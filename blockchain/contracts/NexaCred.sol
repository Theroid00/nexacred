// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title NexaCred Peer-to-Peer Lending Contract
 * @dev Smart contract for decentralized peer-to-peer lending with credit scoring integration
 */
contract NexaCred is ReentrancyGuard, Ownable, Pausable {
    
    // Structs
    struct LoanRequest {
        uint256 id;
        address borrower;
        uint256 amount;
        uint256 interestRate; // Annual interest rate in basis points (100 = 1%)
        uint256 duration; // Loan duration in seconds
        uint256 creditScore;
        uint256 riskScore; // Risk score from ML model (0-1000, where 1000 = 100% risk)
        string purpose;
        LoanStatus status;
        uint256 createdAt;
        uint256 fundedAt;
        uint256 dueDate;
        address lender;
        uint256 totalRepaymentAmount;
        uint256 amountRepaid;
        uint256 lastRepaymentDate;
    }
    
    struct LendingOffer {
        uint256 id;
        address lender;
        uint256 amount;
        uint256 maxInterestRate;
        uint256 minCreditScore;
        uint256 maxRiskScore;
        uint256 maxDuration;
        bool active;
        uint256 createdAt;
    }
    
    struct UserProfile {
        uint256 creditScore;
        uint256 totalBorrowed;
        uint256 totalLent;
        uint256 successfulLoans;
        uint256 defaultedLoans;
        bool kycVerified;
        uint256 reputationScore;
    }
    
    enum LoanStatus {
        PENDING,
        FUNDED,
        REPAID,
        DEFAULTED,
        CANCELLED
    }
    
    // State variables
    uint256 private _loanIdCounter;
    uint256 private _offerIdCounter;
    uint256 public platformFeeRate = 250; // 2.5% in basis points
    uint256 public constant MAX_INTEREST_RATE = 3000; // 30% max annual interest
    uint256 public constant MIN_LOAN_AMOUNT = 0.01 ether;
    uint256 public constant MAX_LOAN_AMOUNT = 100 ether;
    
    mapping(uint256 => LoanRequest) public loans;
    mapping(uint256 => LendingOffer) public offers;
    mapping(address => UserProfile) public userProfiles;
    mapping(address => uint256[]) public userLoans; // borrower -> loan IDs
    mapping(address => uint256[]) public userLendings; // lender -> loan IDs
    mapping(address => bool) public authorizedOracles; // For credit score updates
    
    uint256[] public activeLoanIds;
    uint256[] public activeOfferIds;
    
    // Events
    event LoanRequested(
        uint256 indexed loanId,
        address indexed borrower,
        uint256 amount,
        uint256 interestRate,
        uint256 duration,
        string purpose
    );
    
    event LoanFunded(
        uint256 indexed loanId,
        address indexed lender,
        address indexed borrower,
        uint256 amount
    );
    
    event LoanRepaid(
        uint256 indexed loanId,
        address indexed borrower,
        uint256 amount,
        uint256 totalRepaid
    );
    
    event LoanDefaulted(
        uint256 indexed loanId,
        address indexed borrower,
        uint256 amountLost
    );
    
    event OfferCreated(
        uint256 indexed offerId,
        address indexed lender,
        uint256 amount,
        uint256 maxInterestRate
    );
    
    event CreditScoreUpdated(
        address indexed user,
        uint256 oldScore,
        uint256 newScore
    );
    
    // Modifiers
    modifier onlyAuthorizedOracle() {
        require(authorizedOracles[msg.sender], "Not authorized oracle");
        _;
    }
    
    modifier validLoan(uint256 loanId) {
        require(loanId > 0 && loanId <= _loanIdCounter, "Invalid loan ID");
        _;
    }
    
    modifier validOffer(uint256 offerId) {
        require(offerId > 0 && offerId <= _offerIdCounter, "Invalid offer ID");
        _;
    }
    
    constructor() {
        // Initialize platform
        _loanIdCounter = 0;
        _offerIdCounter = 0;
    }
    
    /**
     * @dev Request a loan
     */
    function requestLoan(
        uint256 amount,
        uint256 interestRate,
        uint256 duration,
        uint256 creditScore,
        uint256 riskScore,
        string calldata purpose
    ) external whenNotPaused nonReentrant {
        require(amount >= MIN_LOAN_AMOUNT && amount <= MAX_LOAN_AMOUNT, "Invalid loan amount");
        require(interestRate <= MAX_INTEREST_RATE, "Interest rate too high");
        require(duration >= 1 days && duration <= 365 days, "Invalid duration");
        require(creditScore >= 300 && creditScore <= 850, "Invalid credit score");
        require(riskScore <= 1000, "Invalid risk score");
        require(bytes(purpose).length > 0, "Purpose required");
        
        _loanIdCounter++;
        uint256 loanId = _loanIdCounter;
        
        // Calculate total repayment amount
        uint256 interest = (amount * interestRate * duration) / (365 days * 10000);
        uint256 totalRepayment = amount + interest;
        
        loans[loanId] = LoanRequest({
            id: loanId,
            borrower: msg.sender,
            amount: amount,
            interestRate: interestRate,
            duration: duration,
            creditScore: creditScore,
            riskScore: riskScore,
            purpose: purpose,
            status: LoanStatus.PENDING,
            createdAt: block.timestamp,
            fundedAt: 0,
            dueDate: 0,
            lender: address(0),
            totalRepaymentAmount: totalRepayment,
            amountRepaid: 0,
            lastRepaymentDate: 0
        });
        
        userLoans[msg.sender].push(loanId);
        activeLoanIds.push(loanId);
        
        // Update user profile
        userProfiles[msg.sender].creditScore = creditScore;
        
        emit LoanRequested(loanId, msg.sender, amount, interestRate, duration, purpose);
    }
    
    /**
     * @dev Fund a loan
     */
    function fundLoan(uint256 loanId) 
        external 
        payable 
        whenNotPaused 
        nonReentrant 
        validLoan(loanId) 
    {
        LoanRequest storage loan = loans[loanId];
        require(loan.status == LoanStatus.PENDING, "Loan not available for funding");
        require(loan.borrower != msg.sender, "Cannot fund your own loan");
        require(msg.value == loan.amount, "Incorrect funding amount");
        
        // Transfer funds to borrower
        uint256 platformFee = (loan.amount * platformFeeRate) / 10000;
        uint256 borrowerAmount = loan.amount - platformFee;
        
        payable(loan.borrower).transfer(borrowerAmount);
        
        // Update loan details
        loan.status = LoanStatus.FUNDED;
        loan.lender = msg.sender;
        loan.fundedAt = block.timestamp;
        loan.dueDate = block.timestamp + loan.duration;
        
        // Update user profiles
        userProfiles[msg.sender].totalLent += loan.amount;
        userProfiles[loan.borrower].totalBorrowed += loan.amount;
        userLendings[msg.sender].push(loanId);
        
        emit LoanFunded(loanId, msg.sender, loan.borrower, loan.amount);
    }
    
    /**
     * @dev Repay a loan (full or partial)
     */
    function repayLoan(uint256 loanId) 
        external 
        payable 
        whenNotPaused 
        nonReentrant 
        validLoan(loanId) 
    {
        LoanRequest storage loan = loans[loanId];
        require(loan.status == LoanStatus.FUNDED, "Loan not active");
        require(loan.borrower == msg.sender, "Only borrower can repay");
        require(msg.value > 0, "Repayment amount must be positive");
        
        uint256 remainingAmount = loan.totalRepaymentAmount - loan.amountRepaid;
        require(msg.value <= remainingAmount, "Overpayment not allowed");
        
        loan.amountRepaid += msg.value;
        loan.lastRepaymentDate = block.timestamp;
        
        // Transfer to lender
        payable(loan.lender).transfer(msg.value);
        
        if (loan.amountRepaid >= loan.totalRepaymentAmount) {
            loan.status = LoanStatus.REPAID;
            userProfiles[loan.borrower].successfulLoans++;
            userProfiles[loan.borrower].reputationScore += 10;
            userProfiles[loan.lender].reputationScore += 5;
        }
        
        emit LoanRepaid(loanId, msg.sender, msg.value, loan.amountRepaid);
    }
    
    /**
     * @dev Mark loan as defaulted (can be called by lender after due date)
     */
    function markAsDefaulted(uint256 loanId) 
        external 
        whenNotPaused 
        validLoan(loanId) 
    {
        LoanRequest storage loan = loans[loanId];
        require(loan.status == LoanStatus.FUNDED, "Loan not active");
        require(loan.lender == msg.sender, "Only lender can mark as defaulted");
        require(block.timestamp > loan.dueDate, "Loan not yet due");
        
        loan.status = LoanStatus.DEFAULTED;
        userProfiles[loan.borrower].defaultedLoans++;
        userProfiles[loan.borrower].reputationScore = userProfiles[loan.borrower].reputationScore > 20 
            ? userProfiles[loan.borrower].reputationScore - 20 : 0;
        
        uint256 amountLost = loan.totalRepaymentAmount - loan.amountRepaid;
        emit LoanDefaulted(loanId, loan.borrower, amountLost);
    }
    
    /**
     * @dev Create a lending offer
     */
    function createLendingOffer(
        uint256 amount,
        uint256 maxInterestRate,
        uint256 minCreditScore,
        uint256 maxRiskScore,
        uint256 maxDuration
    ) external whenNotPaused {
        require(amount >= MIN_LOAN_AMOUNT, "Minimum offer amount not met");
        require(maxInterestRate <= MAX_INTEREST_RATE, "Interest rate too high");
        require(minCreditScore >= 300 && minCreditScore <= 850, "Invalid credit score range");
        require(maxRiskScore <= 1000, "Invalid risk score");
        
        _offerIdCounter++;
        uint256 offerId = _offerIdCounter;
        
        offers[offerId] = LendingOffer({
            id: offerId,
            lender: msg.sender,
            amount: amount,
            maxInterestRate: maxInterestRate,
            minCreditScore: minCreditScore,
            maxRiskScore: maxRiskScore,
            maxDuration: maxDuration,
            active: true,
            createdAt: block.timestamp
        });
        
        activeOfferIds.push(offerId);
        
        emit OfferCreated(offerId, msg.sender, amount, maxInterestRate);
    }
    
    /**
     * @dev Update credit score (only by authorized oracles)
     */
    function updateCreditScore(address user, uint256 newScore) 
        external 
        onlyAuthorizedOracle 
    {
        require(newScore >= 300 && newScore <= 850, "Invalid credit score");
        
        uint256 oldScore = userProfiles[user].creditScore;
        userProfiles[user].creditScore = newScore;
        
        emit CreditScoreUpdated(user, oldScore, newScore);
    }
    
    /**
     * @dev Verify KYC status (only owner)
     */
    function verifyKYC(address user, bool verified) external onlyOwner {
        userProfiles[user].kycVerified = verified;
    }
    
    /**
     * @dev Add/remove authorized oracle (only owner)
     */
    function setAuthorizedOracle(address oracle, bool authorized) external onlyOwner {
        authorizedOracles[oracle] = authorized;
    }
    
    /**
     * @dev Update platform fee rate (only owner)
     */
    function setPlatformFeeRate(uint256 newRate) external onlyOwner {
        require(newRate <= 1000, "Fee rate too high"); // Max 10%
        platformFeeRate = newRate;
    }
    
    /**
     * @dev Withdraw platform fees (only owner)
     */
    function withdrawPlatformFees() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No fees to withdraw");
        payable(owner()).transfer(balance);
    }
    
    /**
     * @dev Pause/unpause contract (only owner)
     */
    function setPaused(bool paused) external onlyOwner {
        if (paused) {
            _pause();
        } else {
            _unpause();
        }
    }
    
    // View functions
    function getLoan(uint256 loanId) external view validLoan(loanId) returns (LoanRequest memory) {
        return loans[loanId];
    }
    
    function getOffer(uint256 offerId) external view validOffer(offerId) returns (LendingOffer memory) {
        return offers[offerId];
    }
    
    function getUserProfile(address user) external view returns (UserProfile memory) {
        return userProfiles[user];
    }
    
    function getUserLoans(address user) external view returns (uint256[] memory) {
        return userLoans[user];
    }
    
    function getUserLendings(address user) external view returns (uint256[] memory) {
        return userLendings[user];
    }
    
    function getActiveLoans() external view returns (uint256[] memory) {
        return activeLoanIds;
    }
    
    function getActiveOffers() external view returns (uint256[] memory) {
        return activeOfferIds;
    }
    
    function getContractStats() external view returns (
        uint256 totalLoans,
        uint256 totalOffers,
        uint256 totalValueLocked
    ) {
        return (_loanIdCounter, _offerIdCounter, address(this).balance);
    }
    
    // Emergency functions
    receive() external payable {
        // Allow contract to receive Ether for platform fees
    }
    
    fallback() external payable {
        revert("Function not found");
    }
}
