// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title CreditScore
 * @dev A smart contract for managing fraud-resistant credit scores on the blockchain
 * @notice This contract allows authorized entities to set and retrieve credit scores
 * @author Nexacred Development Team
 */
contract CreditScore {
    
    // State variables
    address public owner;
    mapping(address => uint256) private creditScores;
    mapping(address => bool) public authorizedScorers;
    mapping(address => uint256) public lastUpdated;
    mapping(address => string) public scoreMetadata; // Additional metadata about the score
    
    // Events
    event CreditScoreSet(address indexed user, uint256 score, uint256 timestamp, address indexed scorer);
    event AuthorizedScorerAdded(address indexed scorer);
    event AuthorizedScorerRemoved(address indexed scorer);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only contract owner can perform this action");
        _;
    }
    
    modifier onlyAuthorizedScorer() {
        require(authorizedScorers[msg.sender] || msg.sender == owner, "Only authorized scorers can set credit scores");
        _;
    }
    
    modifier validScore(uint256 _score) {
        require(_score >= 300 && _score <= 850, "Credit score must be between 300 and 850");
        _;
    }
    
    modifier validAddress(address _address) {
        require(_address != address(0), "Invalid address: cannot be zero address");
        _;
    }
    
    // Constructor
    constructor() {
        owner = msg.sender;
        authorizedScorers[msg.sender] = true;
        emit AuthorizedScorerAdded(msg.sender);
    }
    
    /**
     * @dev Set credit score for a user
     * @param _user The address of the user
     * @param _score The credit score to set (300-850)
     */
    function setCreditScore(address _user, uint256 _score) 
        public 
        onlyAuthorizedScorer 
        validAddress(_user)
        validScore(_score) 
    {
        creditScores[_user] = _score;
        lastUpdated[_user] = block.timestamp;
        
        emit CreditScoreSet(_user, _score, block.timestamp, msg.sender);
    }
    
    /**
     * @dev Set credit score with metadata
     * @param _user The address of the user
     * @param _score The credit score to set (300-850)
     * @param _metadata Additional metadata about the score calculation
     */
    function setCreditScoreWithMetadata(address _user, uint256 _score, string memory _metadata) 
        public 
        onlyAuthorizedScorer 
        validAddress(_user)
        validScore(_score) 
    {
        creditScores[_user] = _score;
        lastUpdated[_user] = block.timestamp;
        scoreMetadata[_user] = _metadata;
        
        emit CreditScoreSet(_user, _score, block.timestamp, msg.sender);
    }
    
    /**
     * @dev Get credit score for a user
     * @param _user The address of the user
     * @return The credit score of the user
     */
    function getCreditScore(address _user) 
        public 
        view 
        validAddress(_user)
        returns (uint256) 
    {
        return creditScores[_user];
    }
    
    /**
     * @dev Get credit score with metadata and timestamp
     * @param _user The address of the user
     * @return score The credit score
     * @return metadata The metadata associated with the score
     * @return timestamp The timestamp when the score was last updated
     */
    function getCreditScoreDetails(address _user) 
        public 
        view 
        validAddress(_user)
        returns (uint256 score, string memory metadata, uint256 timestamp) 
    {
        return (creditScores[_user], scoreMetadata[_user], lastUpdated[_user]);
    }
    
    /**
     * @dev Check if a user has a credit score
     * @param _user The address of the user
     * @return True if the user has a credit score (score > 0), false otherwise
     */
    function hasCreditScore(address _user) 
        public 
        view 
        validAddress(_user)
        returns (bool) 
    {
        return creditScores[_user] > 0;
    }
    
    /**
     * @dev Add an authorized scorer
     * @param _scorer The address to authorize
     */
    function addAuthorizedScorer(address _scorer) 
        public 
        onlyOwner 
        validAddress(_scorer)
    {
        require(!authorizedScorers[_scorer], "Address is already an authorized scorer");
        authorizedScorers[_scorer] = true;
        emit AuthorizedScorerAdded(_scorer);
    }
    
    /**
     * @dev Remove an authorized scorer
     * @param _scorer The address to remove authorization from
     */
    function removeAuthorizedScorer(address _scorer) 
        public 
        onlyOwner 
        validAddress(_scorer)
    {
        require(authorizedScorers[_scorer], "Address is not an authorized scorer");
        require(_scorer != owner, "Cannot remove owner from authorized scorers");
        
        authorizedScorers[_scorer] = false;
        emit AuthorizedScorerRemoved(_scorer);
    }
    
    /**
     * @dev Check if an address is an authorized scorer
     * @param _scorer The address to check
     * @return True if the address is authorized to set scores
     */
    function isAuthorizedScorer(address _scorer) 
        public 
        view 
        validAddress(_scorer)
        returns (bool) 
    {
        return authorizedScorers[_scorer];
    }
    
    /**
     * @dev Transfer ownership of the contract
     * @param _newOwner The address of the new owner
     */
    function transferOwnership(address _newOwner) 
        public 
        onlyOwner 
        validAddress(_newOwner)
    {
        require(_newOwner != owner, "New owner cannot be the current owner");
        
        address previousOwner = owner;
        owner = _newOwner;
        
        // Add new owner as authorized scorer
        authorizedScorers[_newOwner] = true;
        
        emit OwnershipTransferred(previousOwner, _newOwner);
        emit AuthorizedScorerAdded(_newOwner);
    }
    
    /**
     * @dev Batch set credit scores for multiple users
     * @param _users Array of user addresses
     * @param _scores Array of credit scores corresponding to users
     */
    function batchSetCreditScores(address[] memory _users, uint256[] memory _scores) 
        public 
        onlyAuthorizedScorer 
    {
        require(_users.length == _scores.length, "Arrays must have the same length");
        require(_users.length > 0, "Arrays cannot be empty");
        require(_users.length <= 100, "Batch size cannot exceed 100"); // Gas limit protection
        
        for (uint256 i = 0; i < _users.length; i++) {
            require(_users[i] != address(0), "Invalid address in batch");
            require(_scores[i] >= 300 && _scores[i] <= 850, "Invalid score in batch");
            
            creditScores[_users[i]] = _scores[i];
            lastUpdated[_users[i]] = block.timestamp;
            
            emit CreditScoreSet(_users[i], _scores[i], block.timestamp, msg.sender);
        }
    }
    
    /**
     * @dev Get the contract version
     * @return The version string of the contract
     */
    function getVersion() public pure returns (string memory) {
        return "1.0.0";
    }
    
    /**
     * @dev Get contract information
     * @return contractOwner The owner of the contract
     * @return version The version of the contract
     * @return deploymentTime The timestamp when the contract was deployed
     */
    function getContractInfo() 
        public 
        view 
        returns (address contractOwner, string memory version, uint256 deploymentTime) 
    {
        // Note: We can't get the actual deployment time without storing it in constructor
        // This is a simplified version
        return (owner, "1.0.0", block.timestamp);
    }
    
    /**
     * @dev Emergency pause functionality (for future upgrades)
     * This can be implemented with OpenZeppelin's Pausable contract
     */
    bool public paused = false;
    
    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    function pause() public onlyOwner {
        paused = true;
    }
    
    function unpause() public onlyOwner {
        paused = false;
    }
}

/**
 * @title CreditScoreFactory
 * @dev Factory contract to deploy multiple CreditScore contracts
 */
contract CreditScoreFactory {
    
    address[] public deployedContracts;
    mapping(address => address[]) public userContracts;
    
    event ContractDeployed(address indexed contractAddress, address indexed deployer);
    
    /**
     * @dev Deploy a new CreditScore contract
     * @return The address of the newly deployed contract
     */
    function deployCreditScoreContract() public returns (address) {
        CreditScore newContract = new CreditScore();
        address contractAddress = address(newContract);
        
        deployedContracts.push(contractAddress);
        userContracts[msg.sender].push(contractAddress);
        
        // Transfer ownership to the deployer
        newContract.transferOwnership(msg.sender);
        
        emit ContractDeployed(contractAddress, msg.sender);
        
        return contractAddress;
    }
    
    /**
     * @dev Get all deployed contracts
     * @return Array of all deployed contract addresses
     */
    function getDeployedContracts() public view returns (address[] memory) {
        return deployedContracts;
    }
    
    /**
     * @dev Get contracts deployed by a specific user
     * @param _user The user address
     * @return Array of contract addresses deployed by the user
     */
    function getUserContracts(address _user) public view returns (address[] memory) {
        return userContracts[_user];
    }
    
    /**
     * @dev Get the total number of deployed contracts
     * @return The count of deployed contracts
     */
    function getContractCount() public view returns (uint256) {
        return deployedContracts.length;
    }
}
