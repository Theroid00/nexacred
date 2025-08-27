// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title NexaCred Lending Platform
 * @dev Peer-to-peer lending platform with credit score integration
 * 
 * How it works:
 * 1. Borrowers request loans with their credit scores
 * 2. Lenders can fund loans that meet their criteria
 * 3. Borrowers repay loans over time
 * 4. Platform tracks reputation and handles defaults
 * 
 * Backend Integration Points:
 * - requestLoan() when user submits loan application
 * - fundLoan() when lender approves a loan
 * - repayLoan() for processing payments
 * - updateUserCreditScore() from ML service
 * 
 * Frontend Integration Points:
 * - getActiveLoans() to display available loans
 * - getUserProfile() to show user statistics
 * - Event listeners for real-time updates
 */
contract NexaCred {
    // Contract administration
    address public owner;
    bool private reentrancyLock;
    bool public emergencyPause;
    
    // Platform settings
    uint256 public platformFeeRate = 250; // 2.5% (250 basis points)
    uint256 public constant MAX_INTEREST_RATE = 3000; // 30% annual max
    uint256 public constant MIN_LOAN_AMOUNT = 0.01 ether;
    uint256 public constant MAX_LOAN_AMOUNT = 100 ether;
    
    // Loan statuses
    enum LoanStatus { 
        PENDING,    // Waiting for lender
        FUNDED,     // Money transferred, repayment in progress
        REPAID,     // Fully repaid
        DEFAULTED,  // Failed to repay on time
        CANCELLED   // Cancelled by borrower
    }
    
    // Main loan data structure
    struct Loan {
        uint256 id;
        address borrower;
        address lender;
        uint256 amount;
        uint256 interestRate;
        uint256 durationDays;
        uint256 creditScore;
        string purpose;
        LoanStatus status;
        uint256 createdAt;
        uint256 fundedAt;
        uint256 dueDate;
        uint256 totalOwed;
        uint256 amountRepaid;
    }
    
    // Lending offer from investors
    struct LendingOffer {
        uint256 id;
        address lender;
        uint256 amount;
        uint256 maxInterestRate;
        uint256 minCreditScore;
        bool active;
        uint256 createdAt;
    }
    
    // User statistics and reputation
    struct UserProfile {
        uint256 creditScore;
        uint256 totalBorrowed;
        uint256 totalLent;
        uint256 successfulLoans;
        uint256 defaultedLoans;
        bool kycVerified;
        uint256 reputation;
    }
    
    // Contract state
    uint256 private nextLoanId = 1;
    uint256 private nextOfferId = 1;
    
    mapping(uint256 => Loan) public loans;
    mapping(uint256 => LendingOffer) public offers;
    mapping(address => UserProfile) public userProfiles;
    mapping(address => uint256[]) public userLoanIds;
    mapping(address => uint256[]) public userLendingIds;
    mapping(address => bool) public creditOracles;
    
    uint256[] public activeLoanIds;
    uint256[] public activeOfferIds;
    
    // Events for backend/frontend integration
    event LoanRequested(uint256 indexed loanId, address indexed borrower, uint256 amount, string purpose);
    event LoanFunded(uint256 indexed loanId, address indexed lender, address indexed borrower, uint256 amount);
    event RepaymentMade(uint256 indexed loanId, uint256 amount, uint256 remaining);
    event LoanCompleted(uint256 indexed loanId, address indexed borrower);
    event LoanDefaulted(uint256 indexed loanId, uint256 lossAmount);
    event OfferCreated(uint256 indexed offerId, address indexed lender, uint256 amount);
    event CreditScoreUpdated(address indexed user, uint256 newScore);
    
    // Access control modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Owner access required");
        _;
    }
    
    modifier onlyOracle() {
        require(creditOracles[msg.sender] || msg.sender == owner, "Oracle access required");
        _;
    }
    
    modifier noReentrancy() {
        require(!reentrancyLock, "Reentrancy blocked");
        reentrancyLock = true;
        _;
        reentrancyLock = false;
    }
    
    modifier whenActive() {
        require(!emergencyPause, "Platform is paused");
        _;
    }
    
    modifier validLoanId(uint256 loanId) {
        require(loanId > 0 && loanId < nextLoanId, "Invalid loan ID");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        creditOracles[msg.sender] = true;
    }
    
    // Core lending functionality
    
    /**
     * Submit a loan request
     * Called by: Backend API when user applies for loan
     */
    function requestLoan(
        uint256 amount,
        uint256 interestRate,
        uint256 durationDays,
        string calldata purpose
    ) external whenActive returns (uint256 loanId) {
        require(amount >= MIN_LOAN_AMOUNT && amount <= MAX_LOAN_AMOUNT, "Invalid loan amount");
        require(interestRate <= MAX_INTEREST_RATE, "Interest rate too high");
        require(durationDays >= 1 && durationDays <= 365, "Invalid duration");
        require(bytes(purpose).length > 0, "Loan purpose required");
        
        loanId = nextLoanId++;
        uint256 duration = durationDays * 1 days;
        uint256 interest = (amount * interestRate * duration) / (365 days * 10000);
        
        loans[loanId] = Loan({
            id: loanId,
            borrower: msg.sender,
            lender: address(0),
            amount: amount,
            interestRate: interestRate,
            durationDays: durationDays,
            creditScore: userProfiles[msg.sender].creditScore,
            purpose: purpose,
            status: LoanStatus.PENDING,
            createdAt: block.timestamp,
            fundedAt: 0,
            dueDate: 0,
            totalOwed: amount + interest,
            amountRepaid: 0
        });
        
        userLoanIds[msg.sender].push(loanId);
        activeLoanIds.push(loanId);
        
        emit LoanRequested(loanId, msg.sender, amount, purpose);
    }
    
    /**
     * Fund a loan request
     * Called by: Backend API when lender approves loan
     */
    function fundLoan(uint256 loanId) external payable whenActive noReentrancy validLoanId(loanId) {
        Loan storage loan = loans[loanId];
        require(loan.status == LoanStatus.PENDING, "Loan not available");
        require(loan.borrower != msg.sender, "Cannot fund your own loan");
        require(msg.value == loan.amount, "Incorrect funding amount");
        
        // Calculate fees
        uint256 platformFee = (loan.amount * platformFeeRate) / 10000;
        uint256 borrowerReceives = loan.amount - platformFee;
        
        // Update loan details
        loan.lender = msg.sender;
        loan.status = LoanStatus.FUNDED;
        loan.fundedAt = block.timestamp;
        loan.dueDate = block.timestamp + (loan.durationDays * 1 days);
        
        // Update user profiles
        userProfiles[msg.sender].totalLent += loan.amount;
        userProfiles[loan.borrower].totalBorrowed += loan.amount;
        userLendingIds[msg.sender].push(loanId);
        
        // Transfer funds to borrower
        payable(loan.borrower).transfer(borrowerReceives);
        
        emit LoanFunded(loanId, msg.sender, loan.borrower, loan.amount);
    }
    
    /**
     * Make loan repayment
     * Called by: Backend API when borrower makes payment
     */
    function repayLoan(uint256 loanId) external payable whenActive noReentrancy validLoanId(loanId) {
        Loan storage loan = loans[loanId];
        require(loan.status == LoanStatus.FUNDED, "Loan not active");
        require(loan.borrower == msg.sender, "Only borrower can repay");
        require(msg.value > 0, "Payment amount required");
        
        uint256 remaining = loan.totalOwed - loan.amountRepaid;
        require(msg.value <= remaining, "Payment exceeds owed amount");
        
        loan.amountRepaid += msg.value;
        
        // Check if loan is fully repaid
        if (loan.amountRepaid >= loan.totalOwed) {
            loan.status = LoanStatus.REPAID;
            userProfiles[loan.borrower].successfulLoans++;
            userProfiles[loan.borrower].reputation += 10;
            userProfiles[loan.lender].reputation += 5;
            emit LoanCompleted(loanId, loan.borrower);
        }
        
        // Transfer payment to lender
        payable(loan.lender).transfer(msg.value);
        
        emit RepaymentMade(loanId, msg.value, remaining - msg.value);
    }
    
    /**
     * Mark loan as defaulted
     * Called by: Lender after due date passes
     */
    function markDefault(uint256 loanId) external whenActive validLoanId(loanId) {
        Loan storage loan = loans[loanId];
        require(loan.status == LoanStatus.FUNDED, "Loan not active");
        require(loan.lender == msg.sender, "Only lender can mark default");
        require(block.timestamp > loan.dueDate, "Loan not overdue yet");
        
        loan.status = LoanStatus.DEFAULTED;
        userProfiles[loan.borrower].defaultedLoans++;
        
        // Reduce borrower reputation
        if (userProfiles[loan.borrower].reputation >= 20) {
            userProfiles[loan.borrower].reputation -= 20;
        } else {
            userProfiles[loan.borrower].reputation = 0;
        }
        
        uint256 lossAmount = loan.totalOwed - loan.amountRepaid;
        emit LoanDefaulted(loanId, lossAmount);
    }
    
    // Credit score management (for ML service integration)
    
    /**
     * Update user's credit score
     * Called by: Backend ML service after score calculation
     */
    function updateUserCreditScore(address user, uint256 newScore) external onlyOracle {
        require(newScore >= 300 && newScore <= 850, "Credit score must be 300-850");
        userProfiles[user].creditScore = newScore;
        emit CreditScoreUpdated(user, newScore);
    }
    
    // Administrative functions
    
    function setCreditOracle(address oracle, bool authorized) external onlyOwner {
        creditOracles[oracle] = authorized;
    }
    
    function verifyUserKYC(address user) external onlyOwner {
        userProfiles[user].kycVerified = true;
    }
    
    function setPlatformFee(uint256 newRate) external onlyOwner {
        require(newRate <= 1000, "Maximum fee is 10%");
        platformFeeRate = newRate;
    }
    
    function toggleEmergencyPause() external onlyOwner {
        emergencyPause = !emergencyPause;
    }
    
    function withdrawPlatformFees() external onlyOwner {
        require(address(this).balance > 0, "No fees to withdraw");
        payable(owner).transfer(address(this).balance);
    }
    
    // Read-only functions for frontend/backend
    
    function getLoan(uint256 loanId) external view validLoanId(loanId) returns (Loan memory) {
        return loans[loanId];
    }
    
    function getUserProfile(address user) external view returns (UserProfile memory) {
        return userProfiles[user];
    }
    
    function getUserLoans(address user) external view returns (uint256[] memory) {
        return userLoanIds[user];
    }
    
    function getUserLendings(address user) external view returns (uint256[] memory) {
        return userLendingIds[user];
    }
    
    function getActiveLoans() external view returns (uint256[] memory) {
        return activeLoanIds;
    }
    
    function getPlatformStats() external view returns (
        uint256 totalLoans,
        uint256 totalValue,
        uint256 feesCollected
    ) {
        return (nextLoanId - 1, address(this).balance, address(this).balance);
    }
    
    /**
     * Check if user can borrow (has good standing)
     * Called by: Frontend before showing loan form
     */
    function canUserBorrow(address user) external view returns (bool) {
        UserProfile memory profile = userProfiles[user];
        
        // Basic eligibility checks
        if (!profile.kycVerified) return false;
        if (profile.creditScore < 500) return false;
        if (profile.defaultedLoans > profile.successfulLoans) return false;
        
        return true;
    }
    
    /**
     * Get loan eligibility info
     * Called by: Frontend to show borrowing limits
     */
    function getBorrowingLimits(address user) external view returns (
        uint256 maxAmount,
        uint256 suggestedRate,
        bool eligible
    ) {
        UserProfile memory profile = userProfiles[user];
        
        if (!canUserBorrow(user)) {
            return (0, 0, false);
        }
        
        // Calculate limits based on credit score
        if (profile.creditScore >= 750) {
            maxAmount = MAX_LOAN_AMOUNT;
            suggestedRate = 800; // 8%
        } else if (profile.creditScore >= 650) {
            maxAmount = 10 ether;
            suggestedRate = 1200; // 12%
        } else {
            maxAmount = 5 ether;
            suggestedRate = 1800; // 18%
        }
        
        eligible = true;
    }
    
    // Allow contract to receive ETH for fees
    receive() external payable {}
}
