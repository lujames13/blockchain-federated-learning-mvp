"""
client.py

Flower 聯邦學習客戶端實作，整合區塊鏈功能 (適配 flwr==1.17)
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional
import random
import time
import json
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr.common.typing import NDArrays, Config, Scalar

from blockchain_connector import BlockchainConnector

# 定義簡單的 CNN 模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 載入 MNIST 數據集並分片
def load_data(client_id: int, num_clients: int = 10) -> Tuple[DataLoader, DataLoader]:
    """載入 MNIST 數據集並根據客戶端 ID 進行分片"""
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    
    # 載入訓練集
    trainset = MNIST("./data", train=True, download=True, transform=transform)
    
    # 計算分片大小
    n_samples = len(trainset)
    samples_per_client = n_samples // num_clients
    
    # 確定當前客戶端的數據范圍
    start_idx = (client_id - 1) % num_clients * samples_per_client
    end_idx = start_idx + samples_per_client
    
    # 取出此客戶端的數據切片
    indices = list(range(start_idx, end_idx))
    client_trainset = torch.utils.data.Subset(trainset, indices)
    
    # 載入測試集 (所有客戶端使用相同的測試集)
    testset = MNIST("./data", train=False, download=True, transform=transform)
    
    # 創建數據加載器
    trainloader = DataLoader(client_trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    
    return trainloader, testloader

# 訓練模型
def train(net, trainloader, epochs: int, device: torch.device):
    """訓練模型"""
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# 評估模型
def test(net, testloader, device: torch.device) -> Tuple[float, float]:
    """評估模型"""
    net.to(device)
    net.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    
    return test_loss, accuracy

# 將模型參數轉換為列表
def get_parameters(net) -> List[np.ndarray]:
    """將模型參數轉換為 numpy 數組列表"""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# 將參數列表載入模型
def set_parameters(net, parameters: List[np.ndarray]):
    """將參數列表載入模型"""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# 計算模型參數的 SHA-256 哈希值
def compute_model_hash(parameters: List[np.ndarray]) -> str:
    """計算模型參數的哈希值"""
    import hashlib
    # 將所有參數合併為一個字節數組
    all_params = b''
    for param in parameters:
        all_params += param.tobytes()
    # 計算 SHA-256 哈希值
    hash_obj = hashlib.sha256(all_params)
    return hash_obj.hexdigest()

# 自定義 Flower 客戶端，整合區塊鏈功能
class BlockchainFlowerClient(fl.client.NumPyClient):
    """整合區塊鏈的 Flower 客戶端"""
    
    def __init__(self, client_id: int, blockchain_connector: BlockchainConnector, 
                 is_malicious: bool = False, attack_type: str = None):
        """初始化客戶端"""
        self.client_id = client_id
        self.blockchain_connector = blockchain_connector
        self.is_malicious = is_malicious
        self.attack_type = attack_type

        # 設置設備
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {self.device}")
        
        # 載入數據
        self.trainloader, self.testloader = load_data(client_id)
        print(f"已載入數據，訓練集大小: {len(self.trainloader.dataset)}，測試集大小: {len(self.testloader.dataset)}")
        
        # 初始化模型
        self.net = Net()
        
        # 註冊到區塊鏈
        self._register_on_blockchain()
    
    def _register_on_blockchain(self) -> bool:
        """在區塊鏈上註冊客戶端"""
        try:
            success = self.blockchain_connector.register_client()
            if success:
                print(f"客戶端 {self.client_id} 已在區塊鏈上註冊")
            else:
                print(f"客戶端 {self.client_id} 註冊失敗")
                # 添加重試邏輯
                for i in range(3):  # 最多重試 3 次
                    print(f"重試註冊 ({i+1}/3)...")
                    time.sleep(2)  # 等待 2 秒
                    success = self.blockchain_connector.register_client()
                    if success:
                        print(f"客戶端 {self.client_id} 在重試後註冊成功")
                        break
            return success
        except Exception as e:
            print(f"註冊客戶端時發生錯誤: {str(e)}")
            return False
    
    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """獲取客戶端屬性"""
        # 獲取客戶端在區塊鏈上的資訊
        try:
            client_info = self.blockchain_connector.get_client_info(self.client_id)
            return {
                "client_id": str(self.client_id),
                "registered_on_blockchain": True,
                "status": client_info.get("status", 0),
                "contribution_score": client_info.get("contributionScore", 0),
                "device": str(self.device)
            }
        except Exception as e:
            print(f"獲取客戶端屬性時發生錯誤: {str(e)}")
            return {
                "client_id": str(self.client_id),
                "registered_on_blockchain": False,
                "device": str(self.device)
            }
    
    def get_parameters(self, config: Config) -> NDArrays:
        """獲取模型參數"""
        return get_parameters(self.net)
    
    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """訓練模型"""
        # 載入全局模型參數
        set_parameters(self.net, parameters)
        
        # 解析配置
        local_epochs = int(config.get("local_epochs", config.get("epochs", 1)))
        round_id = int(config.get("round", config.get("round_id", 1)))
        
        print(f"開始訓練 - 輪次 {round_id}，本地訓練輪次 {local_epochs}")
        
        # 訓練模型
        train(self.net, self.trainloader, local_epochs, self.device)
        
        # 獲取更新後的參數
        updated_parameters = get_parameters(self.net)
        
        # 如果是惡意客戶端，修改參數
        if self.is_malicious:
            if self.attack_type == "model_poisoning":
                print(f"客戶端 {self.client_id} 執行模型污染攻擊")
                # 對參數進行污染 (例如反轉或添加噪聲)
                for i in range(len(updated_parameters)):
                    # 可以選擇不同的攻擊方式:
                    # 1. 參數反轉
                    updated_parameters[i] = -updated_parameters[i]
                    # 2. 添加大量噪聲
                    # noise = np.random.normal(0, 2.0, updated_parameters[i].shape)
                    # updated_parameters[i] += noise
            
            elif self.attack_type == "targeted_attack":
                print(f"客戶端 {self.client_id} 執行目標攻擊")
                # 執行目標攻擊 (例如讓模型對特定數字分類錯誤)
                # 這需要更複雜的實現...
        
        # 生成模型哈希值
        model_hash = compute_model_hash(updated_parameters)
        
        # 提交模型更新到區塊鏈
        try:
            blockchain_hash = self.blockchain_connector.submit_model_update(round_id, updated_parameters)
            print(f"模型更新已提交到區塊鏈，哈希: {blockchain_hash}")
            if blockchain_hash:
                model_hash = blockchain_hash  # 使用區塊鏈返回的哈希
        except Exception as e:
            print(f"提交模型更新到區塊鏈時發生錯誤: {str(e)}")
            print(f"繼續使用本地生成的模型哈希: {model_hash}")
        
        # 返回更新後的參數和統計數據
        return updated_parameters, len(self.trainloader.dataset), {
            "model_hash": model_hash,
            "client_id": str(self.client_id),
            "round_id": round_id,
            "is_malicious": self.is_malicious
        }
    
    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """評估模型"""
        # 載入參數
        set_parameters(self.net, parameters)
        
        # 評估模型
        loss, accuracy = test(self.net, self.testloader, self.device)
        
        # 獲取輪次 ID
        round_id = int(config.get("round", config.get("round_id", 1)))
        
        print(f"評估完成 - 輪次 {round_id}, 損失: {loss:.4f}, 準確率: {accuracy:.4f}")
        
        # 返回評估結果
        return float(loss), len(self.testloader.dataset), {
            "accuracy": float(accuracy),
            "client_id": str(self.client_id)
        }


# 啟動客戶端
def run_client(client_id: int, blockchain_connector: BlockchainConnector, 
            is_malicious: bool = False, attack_type: str = None,
            server_address: str = "127.0.0.1:8080"):
    """啟動 Flower 客戶端"""
    # 創建客戶端
    client = BlockchainFlowerClient(
        client_id, 
        blockchain_connector,
        is_malicious=is_malicious,
        attack_type=attack_type
    )
    
    # 直接啟動客戶端
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="啟動 Flower 聯邦學習客戶端，整合區塊鏈功能")
    parser.add_argument("--client-id", type=int, required=True, help="客戶端 ID")
    parser.add_argument("--contract-address", type=str, required=True, help="智能合約地址")
    parser.add_argument("--node-url", type=str, default="http://127.0.0.1:8545", help="以太坊節點 URL")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Flower 服務器地址")
    parser.add_argument("--malicious", action="store_true", help="是否為惡意客戶端")
    parser.add_argument("--attack-type", type=str, default="model_poisoning", 
                      choices=["model_poisoning", "targeted_attack"], help="攻擊類型")
    
    args = parser.parse_args()
    
    # 輸出版本資訊
    print(f"Flower 版本: {fl.__version__}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 創建區塊鏈連接器
    blockchain_connector = BlockchainConnector(
        contract_address=args.contract_address,
        client_id=args.client_id,
        node_url=args.node_url
    )
    
    # 運行客戶端
    run_client(
        args.client_id, 
        blockchain_connector, 
        is_malicious=args.malicious,
        attack_type=args.attack_type,
        server_address=args.server_address
    )

