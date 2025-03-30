// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FederatedLearning
 * @dev 管理聯邦學習流程的智能合約
 */
contract FederatedLearning {
    // 合約所有者
    address public owner;

    // 聯邦學習輪次狀態
    enum RoundStatus { Inactive, Active, Completed }

    // 參與者狀態
    enum ClientStatus { Unregistered, Registered, Active, Inactive }
    
    // 參與者結構
    struct Client {
        address clientAddress;
        ClientStatus status;
        uint256 contributionScore;
        uint256 lastUpdateTimestamp;
        bool selectedForRound;
    }
    
    // 輪次結構
    struct Round {
        uint256 roundId;
        RoundStatus status;
        uint256 startTime;
        uint256 endTime;
        uint256 participantCount;
        uint256 completedUpdates;
        string globalModelHash;
        mapping(uint256 => bool) clientParticipation;
    }
    
    // 模型更新結構
    struct ModelUpdate {
        uint256 clientId;
        uint256 roundId;
        string modelUpdateHash;
        uint256 timestamp;
        bool accepted;
    }

    // 儲存參與者資訊
    mapping(uint256 => Client) public clients;
    
    // 儲存輪次資訊
    mapping(uint256 => Round) public rounds;
    
    // 儲存模型更新
    mapping(uint256 => mapping(uint256 => ModelUpdate)) public modelUpdates;
    
    // 全局計數器
    uint256 public clientCount;
    uint256 public roundCount;
    uint256 public currentRoundId;
    
    // 事件宣告
    event ClientRegistered(uint256 indexed clientId, address clientAddress);
    event RoundStarted(uint256 indexed roundId, uint256 startTime);
    event RoundCompleted(uint256 indexed roundId, uint256 endTime, string globalModelHash);
    event ModelUpdateSubmitted(uint256 indexed clientId, uint256 indexed roundId, string modelUpdateHash);
    event ModelUpdateAccepted(uint256 indexed clientId, uint256 indexed roundId);
    event GlobalModelUpdated(uint256 indexed roundId, string globalModelHash);
    event ClientRewarded(uint256 indexed clientId, uint256 roundId, uint256 rewardAmount);

    // 修飾符
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyRegisteredClient(uint256 clientId) {
        require(clients[clientId].status != ClientStatus.Unregistered, "Client not registered");
        _;
    }
    
    modifier onlyActiveRound() {
        require(currentRoundId > 0, "No active round");
        require(rounds[currentRoundId].status == RoundStatus.Active, "Round not active");
        _;
    }

    /**
     * @dev 合約建構子
     */
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev 初始化合約
     */
    function initialize() external onlyOwner {
        require(clientCount == 0 && roundCount == 0, "Already initialized");
        clientCount = 0;
        roundCount = 0;
        currentRoundId = 0;
    }
    
    /**
     * @dev 註冊新的客戶端參與者
     * @param clientId 客戶端 ID
     */
    function registerClient(uint256 clientId) external {
        require(clients[clientId].status == ClientStatus.Unregistered, "Client already registered");
        
        clients[clientId] = Client({
            clientAddress: msg.sender,
            status: ClientStatus.Registered,
            contributionScore: 0,
            lastUpdateTimestamp: block.timestamp,
            selectedForRound: false
        });
        
        clientCount++;
        emit ClientRegistered(clientId, msg.sender);
    }
    
    /**
     * @dev 開始新的訓練輪次
     * @param roundId 輪次 ID
     */
    function startRound(uint256 roundId) external onlyOwner {
        require(roundId > 0, "Invalid round ID");
        require(rounds[roundId].status == RoundStatus.Inactive || rounds[roundId].roundId == 0, "Round already exists");
        
        // 如果有活動的輪次，先結束它
        if (currentRoundId > 0 && rounds[currentRoundId].status == RoundStatus.Active) {
            rounds[currentRoundId].status = RoundStatus.Completed;
            rounds[currentRoundId].endTime = block.timestamp;
        }
        
        Round storage newRound = rounds[roundId];
        newRound.roundId = roundId;
        newRound.status = RoundStatus.Active;
        newRound.startTime = block.timestamp;
        newRound.participantCount = 0;
        newRound.completedUpdates = 0;
        
        currentRoundId = roundId;
        roundCount++;
        
        emit RoundStarted(roundId, block.timestamp);
    }
    
    /**
     * @dev 提交模型更新
     * @param clientId 客戶端 ID
     * @param roundId 輪次 ID
     * @param modelUpdateHash 模型更新的 IPFS 雜湊值
     */
    function submitModelUpdate(uint256 clientId, uint256 roundId, string calldata modelUpdateHash) 
        external 
        onlyRegisteredClient(clientId)
    {
        require(rounds[roundId].status == RoundStatus.Active, "Round not active");
        require(clients[clientId].clientAddress == msg.sender, "Not authorized");
        
        modelUpdates[roundId][clientId] = ModelUpdate({
            clientId: clientId,
            roundId: roundId,
            modelUpdateHash: modelUpdateHash,
            timestamp: block.timestamp,
            accepted: false
        });
        
        if (!rounds[roundId].clientParticipation[clientId]) {
            rounds[roundId].participantCount++;
            rounds[roundId].clientParticipation[clientId] = true;
        }
        
        clients[clientId].lastUpdateTimestamp = block.timestamp;
        
        emit ModelUpdateSubmitted(clientId, roundId, modelUpdateHash);
    }
    
    /**
     * @dev 伺服器接受模型更新
     * @param clientId 客戶端 ID
     * @param roundId 輪次 ID
     */
    function acceptModelUpdate(uint256 clientId, uint256 roundId) 
        external 
        onlyOwner
    {
        require(rounds[roundId].status == RoundStatus.Active, "Round not active");
        require(modelUpdates[roundId][clientId].timestamp > 0, "Update not submitted");
        require(!modelUpdates[roundId][clientId].accepted, "Update already accepted");
        
        modelUpdates[roundId][clientId].accepted = true;
        rounds[roundId].completedUpdates++;
        
        // 增加客戶端的貢獻分數
        clients[clientId].contributionScore += 1;
        
        emit ModelUpdateAccepted(clientId, roundId);
    }
    
    /**
     * @dev 更新全局模型
     * @param roundId 輪次 ID
     * @param globalModelHash 全局模型的 IPFS 雜湊值
     */
    function updateGlobalModel(uint256 roundId, string calldata globalModelHash) 
        external 
        onlyOwner
    {
        require(rounds[roundId].status == RoundStatus.Active, "Round not active");
        
        rounds[roundId].globalModelHash = globalModelHash;
        
        emit GlobalModelUpdated(roundId, globalModelHash);
    }
    
    /**
     * @dev 結束訓練輪次
     * @param roundId 輪次 ID
     */
    function completeRound(uint256 roundId) 
        external 
        onlyOwner
    {
        require(rounds[roundId].status == RoundStatus.Active, "Round not active");
        
        rounds[roundId].status = RoundStatus.Completed;
        rounds[roundId].endTime = block.timestamp;
        
        // 如果這是目前活動的輪次，重置目前輪次 ID
        if (currentRoundId == roundId) {
            currentRoundId = 0;
        }
        
        emit RoundCompleted(roundId, block.timestamp, rounds[roundId].globalModelHash);
    }
    
    /**
     * @dev 獎勵參與者
     * @param clientId 客戶端 ID
     * @param roundId 輪次 ID
     * @param rewardAmount 獎勵金額
     */
    function rewardClient(uint256 clientId, uint256 roundId, uint256 rewardAmount) 
        external 
        onlyOwner
        onlyRegisteredClient(clientId)
    {
        require(rounds[roundId].status == RoundStatus.Completed, "Round not completed");
        require(rounds[roundId].clientParticipation[clientId], "Client did not participate");
        require(modelUpdates[roundId][clientId].accepted, "Client update not accepted");
        
        // 在實際應用中，這裡可以實現代幣轉移邏輯
        
        emit ClientRewarded(clientId, roundId, rewardAmount);
    }
    
    /**
     * @dev 獲取客戶端資訊
     * @param clientId 客戶端 ID
     */
    function getClientInfo(uint256 clientId) 
        external 
        view 
        returns (
            address clientAddress,
            ClientStatus status,
            uint256 contributionScore,
            uint256 lastUpdateTimestamp,
            bool selectedForRound
        )
    {
        Client storage client = clients[clientId];
        return (
            client.clientAddress,
            client.status,
            client.contributionScore,
            client.lastUpdateTimestamp,
            client.selectedForRound
        );
    }
    
    /**
     * @dev 獲取輪次資訊
     * @param roundId 輪次 ID
     */
    function getRoundInfo(uint256 roundId) 
        external 
        view 
        returns (
            uint256 id,
            RoundStatus status,
            uint256 startTime,
            uint256 endTime,
            uint256 participantCount,
            uint256 completedUpdates,
            string memory globalModelHash
        )
    {
        Round storage round = rounds[roundId];
        return (
            round.roundId,
            round.status,
            round.startTime,
            round.endTime,
            round.participantCount,
            round.completedUpdates,
            round.globalModelHash
        );
    }
    
    /**
     * @dev 檢查客戶端是否參與了指定輪次
     * @param clientId 客戶端 ID
     * @param roundId 輪次 ID
     */
    function didClientParticipate(uint256 clientId, uint256 roundId) 
        external 
        view 
        returns (bool)
    {
        return rounds[roundId].clientParticipation[clientId];
    }
    
    /**
     * @dev 獲取模型更新資訊
     * @param clientId 客戶端 ID
     * @param roundId 輪次 ID
     */
    function getModelUpdate(uint256 clientId, uint256 roundId) 
        external 
        view 
        returns (
            uint256 cId,
            uint256 rId,
            string memory modelUpdateHash,
            uint256 timestamp,
            bool accepted
        )
    {
        ModelUpdate storage update = modelUpdates[roundId][clientId];
        return (
            update.clientId,
            update.roundId,
            update.modelUpdateHash,
            update.timestamp,
            update.accepted
        );
    }

    /**
     * @dev 獲取系統狀態概覽
     */
    function getSystemStatus() 
        external 
        view 
        returns (
            uint256 totalClients,
            uint256 totalRounds,
            uint256 currentRound,
            RoundStatus currentRoundStatus
        )
    {
        RoundStatus status = RoundStatus.Inactive;
        if (currentRoundId > 0) {
            status = rounds[currentRoundId].status;
        }
        
        return (
            clientCount,
            roundCount,
            currentRoundId,
            status
        );
    }
}