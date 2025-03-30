# 區塊鏈基於 Flower 的聯邦學習 MVP

本專案是一個最小可行產品 (MVP)，展示如何將 Hardhat 區塊鏈框架與 Flower 聯邦學習框架整合。這個實作允許在區塊鏈上管理聯邦學習過程、記錄模型更新並追蹤參與者貢獻。

## 前置需求

- Node.js (v16+)
- Python (v3.8+)
- npm 或 yarn
- pip

## 安裝步驟

1. **複製專案**
   ```bash
   git clone https://github.com/lujames13/blockchain-federated-learning-mvp.git
   cd blockchain-federated-learning-mvp
   ```

2. **安裝 Hardhat 相關依賴**
   ```bash
   npm install
   npm install --save-dev @nomicfoundation/hardhat-ignition-ethers
   ```

3. **安裝 Python 相關依賴**
   ```bash
   pip install -r requirements.txt
   ```

## 專案結構

```
blockchain-federated-learning-mvp/
├── contracts/                      # 智能合約
│   └── FederatedLearning.sol       # 聯邦學習管理合約
├── ignition/                       # Hardhat Ignition 部署模組
│   ├── modules/                    
│   │   └── federated-learning.js   # 合約部署模組
│   └── deployments/                # 部署狀態記錄 (自動生成)
├── scripts/                        # 互動腳本
│   └── interact.js                 # 合約互動功能
├── fl/                             # 聯邦學習相關代碼
│   ├── server.py                   # Flower 伺服器
│   ├── client.py                   # Flower 客戶端
│   └── blockchain_connector.py     # 區塊鏈連接器
├── test/                           # 測試
│   └── FederatedLearning.test.js   # 合約測試
├── hardhat.config.js               # Hardhat 配置
├── requirements.txt                # Python 依賴
└── README.md                       # 本文件
```

## 設定與部署

1. **啟動本地 Hardhat 節點**
   ```bash
   npx hardhat node
   ```

2. **使用 Ignition 部署智能合約**
   ```bash
   npx hardhat ignition deploy ignition/modules/federated-learning.js --network localhost
   ```
   
   部署完成後，合約地址會顯示在終端機輸出中，同時也會儲存在 `ignition/deployments/` 目錄下。

3. **記錄部署的合約地址**
   從部署輸出中獲取合約地址，並將其更新到 `fl/blockchain_connector.py` 中。

## 與合約互動

使用互動腳本可以手動測試已部署的智能合約：

```bash
# 獲取當前聯邦學習狀態
npx hardhat run scripts/interact.js --func getStatus --network localhost

# 註冊新參與者
npx hardhat run scripts/interact.js --func registerClient --clientId 1 --network localhost

# 開始新一輪訓練
npx hardhat run scripts/interact.js --func startRound --roundId 1 --network localhost
```

## 執行聯邦學習

1. **啟動 Flower 伺服器**
   ```bash
   python fl/server.py --contract-address 0x5FbDB2315678afecb367f032d93F642f64180aa3
   ```

2. **啟動多個 Flower 客戶端**
   在不同終端窗口中，啟動多個客戶端：
   ```bash
   python fl/client.py --client-id 1 --contract-address 0x5FbDB2315678afecb367f032d93F642f64180aa3
   ```
   ```bash
   python fl/client.py --client-id 2 --contract-address 0x5FbDB2315678afecb367f032d93F642f64180aa3
   ```

## 系統運作說明

1. **初始設定**:
   - Hardhat 節點啟動一個本地區塊鏈
   - 聯邦學習智能合約通過 Ignition 部署到區塊鏈
   - 合約記錄參與者及其初始狀態

2. **聯邦學習過程**:
   - Flower 伺服器協調學習過程
   - 客戶端執行本地訓練並通過區塊鏈提交更新
   - 伺服器聚合模型並通過區塊鏈註冊新全局模型

3. **區塊鏈整合點**:
   - 參與者註冊管理
   - 模型參數更新記錄
   - 參與者貢獻的驗證與追蹤
   - 激勵機制 (簡化版)

## 開發者指南

### Ignition 模組設計

`ignition/modules/federated-learning.js` 模組負責合約的部署和初始設定：

```javascript
import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

export default buildModule("FederatedLearningModule", (m) => {
  // 部署 FederatedLearning 合約
  const federatedLearning = m.contract("FederatedLearning", []);
  
  // 初始化合約
  m.call(federatedLearning, "initialize", []);
  
  // 其他初始設定
  
  return { federatedLearning };
});
```

### 互動腳本功能

`scripts/interact.js` 提供多種合約互動功能，可以用來：

- 獲取系統狀態
- 管理參與者
- 控制訓練輪次
- 處理模型更新
- 分配獎勵

## 自訂與擴展

1. **修改智能合約**:
   編輯 `contracts/FederatedLearning.sol` 以添加更多功能

2. **更新部署模組**:
   修改 `ignition/modules/federated-learning.js` 以調整部署邏輯

3. **添加互動功能**:
   擴展 `scripts/interact.js` 以支援新的合約功能

4. **自訂聯邦學習邏輯**:
   修改 `fl/server.py` 和 `fl/client.py` 中的模型和訓練邏輯

5. **整合現有模型**:
   將您現有的 PyTorch 或 TensorFlow 模型整合到 `fl/client.py` 中

## 部署到測試網或主網

1. **配置網絡信息**:
   在 `hardhat.config.js` 中設置測試網或主網的配置

2. **部署到目標網絡**:
   ```bash
   npx hardhat ignition deploy ignition/modules/federated-learning.js --network goerli
   ```

3. **更新連接器配置**:
   修改 `fl/blockchain_connector.py` 以連接到相應網絡

## 授權

MIT
