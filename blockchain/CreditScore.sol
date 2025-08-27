// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title CreditScore Registry
 * @dev Stores and manages credit scores on blockchain with complete audit trail
 * 
 * This contract handles:
 * - Secure credit score storage for users
 * - Historical tracking of all score changes
 * - Admin controls for authorized score updates
 * - Fraud detection and flagging system
 * 
 * Integration Points:
 * - Backend API can read scores via getCreditScore()
 * - ML service updates scores via updateScore()
 * - Frontend displays score history from audit trail
 */
contract CreditScore {
    // Contract owner and access control
    address public owner;
    mapping(address => bool) public authorizedUpdaters;

    // Core credit score data
    mapping(address => uint256) private creditScores;
    mapping(address => uint256) public lastUpdated;
    mapping(address => string) public scoreMetadata;

    // Audit trail for transparency
    struct ScoreChange {
        uint256 oldScore;
        uint256 newScore;
        uint256 timestamp;
        address updatedBy;
        string reason;
    }
    mapping(address => ScoreChange[]) private scoreHistory;

    // Fraud protection system
    mapping(address => bool) private fraudFlag;
    struct FraudEvent {
        bool flagged;
        uint256 timestamp;
        address flaggedBy;
        string reason;
    }
    mapping(address => FraudEvent[]) private fraudHistory;

    // ---------------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------------
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event UpdaterAuthorized(address indexed updater, bool authorized);

    event ScoreUpdated(
        address indexed user,
        uint256 oldScore,
        uint256 newScore,
        uint256 timestamp,
        address indexed actor,
        string reason
    );

    event FraudFlagSet(
        address indexed user,
        bool indexed flagged,
        uint256 timestamp,
        address indexed actor,
        string reason
    );

    // Backwards-compat events for existing integrations
    event CreditScoreSet(address indexed user, uint256 score, uint256 timestamp, address indexed scorer);
    event AuthorizedScorerAdded(address indexed scorer);
    event AuthorizedScorerRemoved(address indexed scorer);

    // ---------------------------------------------------------------------
    // Modifiers
    // ---------------------------------------------------------------------
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyAuthorizedUpdater() {
        require(msg.sender == owner || authorizedUpdaters[msg.sender], "Not authorized updater");
        _;
    }

    modifier validAddress(address _addr) {
        require(_addr != address(0), "Zero address");
        _;
    }

    modifier validScore(uint256 _score) {
        require(_score >= 300 && _score <= 850, "Score out of range");
        _;
    }

    // ---------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------
    constructor() {
        owner = msg.sender;
        authorizedUpdaters[msg.sender] = true; // bootstrapped
        emit UpdaterAuthorized(msg.sender, true);
        // Backwards-compat event
        emit AuthorizedScorerAdded(msg.sender);
    }

    function transferOwnership(address _newOwner)
        public
        onlyOwner
        validAddress(_newOwner)
    {
        require(_newOwner != owner, "Same owner");
        address prev = owner;
        owner = _newOwner;
        authorizedUpdaters[_newOwner] = true; // convenience
        emit OwnershipTransferred(prev, _newOwner);
        emit UpdaterAuthorized(_newOwner, true);
        // Backwards-compat event
        emit AuthorizedScorerAdded(_newOwner);
    }

    // ---------------------------------------------------------------------
    // Role management
    // ---------------------------------------------------------------------
    function setAuthorizedUpdater(address _updater, bool _authorized)
        external
        onlyOwner
        validAddress(_updater)
    {
        authorizedUpdaters[_updater] = _authorized;
        emit UpdaterAuthorized(_updater, _authorized);
        // Backwards-compat events for existing tools
        if (_authorized) {
            emit AuthorizedScorerAdded(_updater);
        } else {
            emit AuthorizedScorerRemoved(_updater);
        }
    }

    function isAuthorizedUpdater(address _updater)
        external
        view
        returns (bool)
    {
        return authorizedUpdaters[_updater];
    }

    // ---------------------------------------------------------------------
    // Core: Credit Score Management
    // ---------------------------------------------------------------------
    /**
     * @notice Update the user's credit score (admin/oracle-controlled).
     * @dev Emits ScoreUpdated and legacy CreditScoreSet for compatibility.
     * @param _user User address whose score is updated
     * @param _newScore New score in [300, 850]
     * @param _reason Short reason or model/version identifier
     */
    function updateScore(address _user, uint256 _newScore, string memory _reason)
        public
        onlyAuthorizedUpdater
        validAddress(_user)
        validScore(_newScore)
    {
        uint256 old = creditScores[_user];
        creditScores[_user] = _newScore;
        lastUpdated[_user] = block.timestamp;
        scoreMetadata[_user] = _reason; // store latest reason as metadata for convenience

        // Record change in audit trail
        scoreHistory[_user].push(
            ScoreChange({
                oldScore: old,
                newScore: _newScore,
                timestamp: block.timestamp,
                updatedBy: msg.sender,
                reason: _reason
            })
        );

        emit ScoreUpdated(_user, old, _newScore, block.timestamp, msg.sender, _reason);
        // legacy event many apps might listen to
        emit CreditScoreSet(_user, _newScore, block.timestamp, msg.sender);
    }

    // Backwards-compat function names for existing code (aliases)
    function setCreditScore(address _user, uint256 _score)
        public
        onlyAuthorizedUpdater
        validAddress(_user)
        validScore(_score)
    {
        updateScore(_user, _score, "");
    }

    function setCreditScoreWithMetadata(address _user, uint256 _score, string calldata _metadata)
        public
        onlyAuthorizedUpdater
        validAddress(_user)
        validScore(_score)
    {
        updateScore(_user, _score, _metadata);
    }

    /**
     * @notice Read-only getters used by frontend/backend.
     */
    function getCreditScore(address _user)
        public
        view
        validAddress(_user)
        returns (uint256)
    {
        return creditScores[_user];
    }

    function getCreditScoreDetails(address _user)
        public
        view
        validAddress(_user)
        returns (uint256 score, string memory metadata, uint256 timestamp)
    {
        return (creditScores[_user], scoreMetadata[_user], lastUpdated[_user]);
    }

    function hasCreditScore(address _user)
        public
        view
        validAddress(_user)
        returns (bool)
    {
        return creditScores[_user] > 0;
    }

    // ---------------------------------------------------------------------
    // Audit Trail (score history)
    // ---------------------------------------------------------------------
    function getScoreChangeCount(address _user) external view returns (uint256) {
        return scoreHistory[_user].length;
    }

    function getScoreChange(address _user, uint256 _index)
        external
        view
        returns (
            uint256 oldScore,
            uint256 newScore,
            uint256 timestamp,
            address updatedBy,
            string memory reason
        )
    {
        require(_index < scoreHistory[_user].length, "Index out of bounds");
        ScoreChange storage ch = scoreHistory[_user][_index];
        return (ch.oldScore, ch.newScore, ch.timestamp, ch.updatedBy, ch.reason);
    }

    function getLatestScoreChange(address _user)
        external
        view
        returns (
            bool exists,
            uint256 oldScore,
            uint256 newScore,
            uint256 timestamp,
            address updatedBy,
            string memory reason
        )
    {
        uint256 len = scoreHistory[_user].length;
        if (len == 0) {
            return (false, 0, 0, 0, address(0), "");
        }
        ScoreChange storage ch = scoreHistory[_user][len - 1];
        return (true, ch.oldScore, ch.newScore, ch.timestamp, ch.updatedBy, ch.reason);
    }

    // ---------------------------------------------------------------------
    // Fraud flagging
    // ---------------------------------------------------------------------
    /**
     * @notice Flag or clear a user for fraud. Admin-controlled. Emits immutable history events.
     * @param _user User address
     * @param _flag True to flag as fraud, false to clear
     * @param _reason Reason for flagging/clearing
     */
    function flagFraud(address _user, bool _flag, string calldata _reason)
        external
        onlyOwner
        validAddress(_user)
    {
        fraudFlag[_user] = _flag;
        fraudHistory[_user].push(
            FraudEvent({ flagged: _flag, timestamp: block.timestamp, flaggedBy: msg.sender, reason: _reason })
        );
        emit FraudFlagSet(_user, _flag, block.timestamp, msg.sender, _reason);
    }

    function isFraud(address _user) external view returns (bool) {
        return fraudFlag[_user];
    }

    function getFraudEventsCount(address _user) external view returns (uint256) {
        return fraudHistory[_user].length;
    }

    function getFraudEvent(address _user, uint256 _index)
        external
        view
        returns (bool flagged, uint256 timestamp, address flaggedBy, string memory reason)
    {
        require(_index < fraudHistory[_user].length, "Index out of bounds");
        FraudEvent storage ev = fraudHistory[_user][_index];
        return (ev.flagged, ev.timestamp, ev.flaggedBy, ev.reason);
    }

    // ---------------------------------------------------------------------
    // Batch updates (gas-capped)
    // ---------------------------------------------------------------------
    function batchUpdateScores(address[] calldata _users, uint256[] calldata _scores, string[] calldata _reasons)
        external
        onlyAuthorizedUpdater
    {
        uint256 n = _users.length;
        require(n == _scores.length && n == _reasons.length, "Array length mismatch");
        require(n > 0 && n <= 100, "Invalid batch size");
        for (uint256 i = 0; i < n; i++) {
            require(_users[i] != address(0), "Zero address");
            require(_scores[i] >= 300 && _scores[i] <= 850, "Score out of range");
            updateScore(_users[i], _scores[i], _reasons[i]);
        }
    }

    // ---------------------------------------------------------------------
    // Minimal pause switches (optional extension points for future OZ Pausable)
    // ---------------------------------------------------------------------
    bool public paused = false;

    modifier whenNotPaused() {
        require(!paused, "Paused");
        _;
    }

    function pause() public onlyOwner { paused = true; }
    function unpause() public onlyOwner { paused = false; }
}

/**
 * @title CreditScoreFactory
 * @dev Factory contract to deploy multiple CreditScore contracts
 */
contract CreditScoreFactory {
    address[] public deployedContracts;
    mapping(address => address[]) public userContracts;

    event ContractDeployed(address indexed contractAddress, address indexed deployer);

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

    function getDeployedContracts() public view returns (address[] memory) {
        return deployedContracts;
    }

    function getUserContracts(address _user) public view returns (address[] memory) {
        return userContracts[_user];
    }

    function getContractCount() public view returns (uint256) {
        return deployedContracts.length;
    }
}
