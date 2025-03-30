"""
blockchain_connector.py

區塊鏈連接器模組，負責 Flower 與以太坊智能合約之間的交互
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import time

from web3 import Web3
import numpy as np
from web3.middleware import geth_poa_middleware


class BlockchainConnector:
    """連接聯邦學習系統與區塊鏈的工具"""

    def __init__(
        self, 
        contract_address: str,
        client_id: Optional[int] = None,
        private_key: Optional[str] = None,
        node_url: str = "http://127.0.0.1:8545",
        contract_abi_path: Optional[str] = None,
    ):
        """
        初始化區塊鏈連接器
        
        參數:
            contract_address: 已部署的 FederatedLearning 合約地址
            client_id: 客戶端 ID (用於客戶端)
            private_key: 私鑰 (用於客戶端)
            node_url: 以太坊節點 URL
            contract_abi_path: 合約 ABI 檔案路徑
        """
        self.client_id = client_id
        self.contract_address = contract_address
        
        # 連接到以太坊節點
        self.w3 = Web3(Web3.HTTPProvider(node_url))
        # 增加 PoA 中間件支援 Hardhat
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # 檢查連接
        if not self.w3.is_connected():
            raise ConnectionError(f"無法連接到以太坊節點: {node_url}")
        
        # 設定帳戶
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
            print(f"使用帳戶: {self.account.address}")
        else:
            self.account = self.w3.eth.accounts[0]
            print(f"使用默認帳戶: {self.account}")
        
        # 載入合約 ABI
        if contract_abi_path is None:
            # 嘗試從預設位置載入 ABI
            try:
                # 尋找最新的部署資料夾
                ignition_dir = Path(__file__).parent.parent / "ignition" / "deployments"
                if ignition_dir.exists():
                    deployment_folders = sorted([d for d in ignition_dir.iterdir() if d.is_dir()])
                    if deployment_folders:
                        latest_deployment = deployment_folders[-1]
                        artifacts_path = latest_deployment / "artifacts.json"
                        if artifacts_path.exists():
                            with open(artifacts_path, "r") as file:
                                artifacts = json.load(file)
                                contract_abi = artifacts["contracts"]["FederatedLearning"]["abi"]
                        else:
                            raise FileNotFoundError(f"找不到 artifacts.json 檔案: {artifacts_path}")
                    else:
                        raise FileNotFoundError(f"找不到部署資料夾: {ignition_dir}")
                else:
                    # 直接載入 Artifacts 檔案
                    artifacts_path = Path(__file__).parent.parent / "artifacts" / "contracts" / "FederatedLearning.sol" / "FederatedLearning.json"
                    if artifacts_path.exists():
                        with open(artifacts_path, "r") as file:
                            artifact = json.load(file)
                            contract_abi = artifact["abi"]
                    else:
                        raise FileNotFoundError(f"找不到合約 ABI 檔案: {artifacts_path}")
            except Exception as e:
                raise ValueError(f"無法載入合約 ABI: {str(e)}")
        else:
            # 使用指定的 ABI
            with open(contract_abi_path, "r") as file:
                contract_abi = json.load(file)
        
        # 初始化合約接口
        self.contract = self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=contract_abi)
        print(f"已連接到合約: {contract_address}")
    
    def _get_transaction_params(self) -> Dict:
        """獲取交易參數"""
        return {
            "from": self.account.address if hasattr(self.account, "address") else self.account,
            "gasPrice": self.w3.eth.gas_price,
            "nonce": self.w3.eth.get_transaction_count(
                self.account.address if hasattr(self.account, "address") else self.account
            ),
        }
    
    def _sign_and_send_transaction(self, transaction) -> str:
        """簽署並發送交易"""
        # 估算 gas
        gas_estimate = self.w3.eth.estimate_gas(transaction)
        transaction["gas"] = int(gas_estimate * 1.2)  # 增加 20% 以防止 gas 不足
        
        # 簽署交易 (如果有私鑰)
        if hasattr(self.account, "key"):
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key=self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        else:
            # 使用解鎖的帳戶
            tx_hash = self.w3.eth.send_transaction(transaction)
        
        # 等待交易確認
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt["transactionHash"].hex()
    
    def hash_model_parameters(self, parameters: List[np.ndarray]) -> str:
        """將模型參數轉換為雜湊值"""
        # 將參數序列化為二進制格式
        serialized = b""
        for param in parameters:
            serialized += param.tobytes()
        
        # 計算 SHA-256 雜湊值
        hash_object = hashlib.sha256(serialized)
        return hash_object.hexdigest()
    
    def register_client(self) -> bool:
        """註冊客戶端"""
        if self.client_id is None:
            raise ValueError("客戶端 ID 未設定")
        
        # 建立交易
        transaction = self.contract.functions.registerClient(self.client_id).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            print(f"客戶端 {self.client_id} 註冊成功，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            print(f"註冊客戶端失敗: {str(e)}")
            return False
    
    def start_round(self, round_id: int) -> bool:
        """開始新的訓練輪次"""
        # 建立交易
        transaction = self.contract.functions.startRound(round_id).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            print(f"輪次 {round_id} 開始成功，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            print(f"開始輪次失敗: {str(e)}")
            return False
    
    def submit_model_update(self, round_id: int, parameters: List[np.ndarray]) -> str:
        """提交模型更新"""
        if self.client_id is None:
            raise ValueError("客戶端 ID 未設定")
        
        # 計算模型參數的雜湊值
        model_hash = self.hash_model_parameters(parameters)
        
        # 建立交易
        transaction = self.contract.functions.submitModelUpdate(
            self.client_id, round_id, model_hash
        ).build_transaction(self._get_transaction_params())
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            print(f"客戶端 {self.client_id} 提交模型更新成功，輪次: {round_id}，交易雜湊: {tx_hash}")
            return model_hash
        except Exception as e:
            print(f"提交模型更新失敗: {str(e)}")
            return ""
    
    def accept_model_update(self, client_id: int, round_id: int) -> bool:
        """接受模型更新"""
        # 建立交易
        transaction = self.contract.functions.acceptModelUpdate(
            client_id, round_id
        ).build_transaction(self._get_transaction_params())
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            print(f"接受客戶端 {client_id} 輪次 {round_id} 的模型更新成功，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            print(f"接受模型更新失敗: {str(e)}")
            return False
    
    def update_global_model(self, round_id: int, parameters: List[np.ndarray]) -> str:
        """更新全局模型"""
        # 計算模型參數的雜湊值
        model_hash = self.hash_model_parameters(parameters)
        
        # 建立交易
        transaction = self.contract.functions.updateGlobalModel(
            round_id, model_hash
        ).build_transaction(self._get_transaction_params())
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            print(f"更新輪次 {round_id} 的全局模型成功，交易雜湊: {tx_hash}")
            return model_hash
        except Exception as e:
            print(f"更新全局模型失敗: {str(e)}")
            return ""
    
    def complete_round(self, round_id: int) -> bool:
        """完成訓練輪次"""
        # 建立交易
        transaction = self.contract.functions.completeRound(round_id).build_transaction(
            self._get_transaction_params()
        )
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            print(f"輪次 {round_id} 完成成功，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            print(f"完成輪次失敗: {str(e)}")
            return False
    
    def reward_client(self, client_id: int, round_id: int, reward_amount: int) -> bool:
        """獎勵客戶端"""
        # 建立交易
        transaction = self.contract.functions.rewardClient(
            client_id, round_id, reward_amount
        ).build_transaction(self._get_transaction_params())
        
        try:
            tx_hash = self._sign_and_send_transaction(transaction)
            print(f"獎勵客戶端 {client_id} 輪次 {round_id} 成功，獎勵: {reward_amount}，交易雜湊: {tx_hash}")
            return True
        except Exception as e:
            print(f"獎勵客戶端失敗: {str(e)}")
            return False
    
    def get_client_info(self, client_id: int) -> Dict:
        """獲取客戶端資訊"""
        try:
            client_info = self.contract.functions.getClientInfo(client_id).call()
            return {
                "clientAddress": client_info[0],
                "status": client_info[1],
                "contributionScore": client_info[2],
                "lastUpdateTimestamp": client_info[3],
                "selectedForRound": client_info[4],
            }
        except Exception as e:
            print(f"獲取客戶端資訊失敗: {str(e)}")
            return {}
    
    def get_round_info(self, round_id: int) -> Dict:
        """獲取輪次資訊"""
        try:
            round_info = self.contract.functions.getRoundInfo(round_id).call()
            return {
                "roundId": round_info[0],
                "status": round_info[1],
                "startTime": round_info[2],
                "endTime": round_info[3],
                "participantCount": round_info[4],
                "completedUpdates": round_info[5],
                "globalModelHash": round_info[6],
            }
        except Exception as e:
            print(f"獲取輪次資訊失敗: {str(e)}")
            return {}
    
    def get_system_status(self) -> Dict:
        """獲取系統狀態"""
        try:
            status = self.contract.functions.getSystemStatus().call()
            return {
                "totalClients": status[0],
                "totalRounds": status[1],
                "currentRound": status[2],
                "currentRoundStatus": status[3],
            }
        except Exception as e:
            print(f"獲取系統狀態失敗: {str(e)}")
            return {}
    
    def did_client_participate(self, client_id: int, round_id: int) -> bool:
        """檢查客戶端是否參與輪次"""
        try:
            return self.contract.functions.didClientParticipate(client_id, round_id).call()
        except Exception as e:
            print(f"檢查客戶端參與狀態失敗: {str(e)}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="區塊鏈連接器測試工具")
    parser.add_argument("--contract-address", type=str, required=True, help="FederatedLearning 合約地址")
    parser.add_argument("--client-id", type=int, default=None, help="客戶端 ID")
    parser.add_argument("--node-url", type=str, default="http://127.0.0.1:8545", help="以太坊節點 URL")
    parser.add_argument("--action", type=str, required=True, 
                        choices=["register", "status", "start-round", "complete-round"], 
                        help="要執行的操作")
    parser.add_argument("--round-id", type=int, default=None, help="輪次 ID")
    
    args = parser.parse_args()
    
    connector = BlockchainConnector(
        contract_address=args.contract_address,
        client_id=args.client_id,
        node_url=args.node_url
    )
    
    if args.action == "register":
        if args.client_id is None:
            print("註冊操作需要指定 --client-id")
            exit(1)
        connector.register_client()
    
    elif args.action == "status":
        status = connector.get_system_status()
        print("系統狀態:", json.dumps(status, indent=2))
    
    elif args.action == "start-round":
        if args.round_id is None:
            print("開始輪次操作需要指定 --round-id")
            exit(1)
        connector.start_round(args.round_id)
    
    elif args.action == "complete-round":
        if args.round_id is None:
            print("完成輪次操作需要指定 --round-id")
            exit(1)
        connector.complete_round(args.round_id)