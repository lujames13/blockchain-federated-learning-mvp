"""
server.py

Flower 聯邦學習伺服器實作，整合區塊鏈功能
"""

import argparse
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import time
import json

import flwr as fl
import numpy as np
from flwr.common import (
    Code,
    EvaluateIns, 
    EvaluateRes, 
    FitIns, 
    FitRes, 
    Parameters, 
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from blockchain_connector import BlockchainConnector


# 自定義聯邦學習策略，整合區塊鏈功能
class BlockchainFedAvg(FedAvg):
    """整合區塊鏈的聯邦學習策略"""

    def __init__(
        self,
        blockchain_connector: BlockchainConnector,
        round_id: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blockchain_connector = blockchain_connector
        self.round_id = round_id
        self.client_updates = {}
        
        # 啟動新輪次
        print(f"開始區塊鏈輪次 {round_id}")
        self.blockchain_connector.start_round(round_id)
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """配置客戶端訓練"""
        # 獲取參數數組
        parameter_arrays = parameters_to_ndarrays(parameters)
        
        # 存儲全局模型
        if server_round > 1:  # 第一輪不需要更新全局模型
            model_hash = self.blockchain_connector.update_global_model(
                self.round_id, parameter_arrays
            )
            print(f"更新輪次 {self.round_id} 的全局模型: {model_hash}")
        
        # 呼叫父類方法繼續
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """聚合客戶端更新"""
        # 記錄所有參與此輪訓練的客戶端
        for client, fit_res in results:
            # 從客戶端屬性中獲取客戶端 ID
            client_id = int(client.cid)
            
            # 獲取模型更新雜湊
            properties = fit_res.metrics.get("properties", {})
            if "model_hash" in properties:
                model_hash = properties["model_hash"]
                self.client_updates[client_id] = model_hash
                
                # 接受模型更新
                print(f"接受客戶端 {client_id} 的模型更新，雜湊: {model_hash}")
                self.blockchain_connector.accept_model_update(client_id, self.round_id)
        
        # 呼叫父類方法進行聚合
        return super().aggregate_fit(server_round, results, failures)
    
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """配置客戶端評估"""
        # 呼叫父類方法
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """聚合評估結果"""
        # 呼叫父類方法
        return super().aggregate_evaluate(server_round, results, failures)
    
    def finalize_round(
        self,
        server_round: int,
        parameters: Parameters,
        metrics: Dict[str, Scalar]
    ) -> None:
        """完成輪次"""
        super().finalize_round(server_round, parameters, metrics)
        
        if server_round == self.num_rounds:
            # 最後一輪，完成區塊鏈輪次
            print(f"完成區塊鏈輪次 {self.round_id}")
            # 更新最終全局模型
            parameter_arrays = parameters_to_ndarrays(parameters)
            model_hash = self.blockchain_connector.update_global_model(
                self.round_id, parameter_arrays
            )
            print(f"最終全局模型雜湊: {model_hash}")
            # 完成輪次
            self.blockchain_connector.complete_round(self.round_id)
            
            # 獎勵參與者
            for client_id, _ in self.client_updates.items():
                print(f"獎勵客戶端 {client_id}")
                self.blockchain_connector.reward_client(client_id, self.round_id, 100)


# 自定義客戶端管理器
class BlockchainClientManager(fl.server.ClientManager):
    """整合區塊鏈的客戶端管理器"""
    
    def __init__(self):
        super().__init__()
        self.registered_clients = {}
    
    def register(self, client: ClientProxy) -> bool:
        """註冊新客戶端"""
        # 從客戶端屬性中獲取區塊鏈註冊狀態
        properties = client.properties
        registered_on_blockchain = properties.get("registered_on_blockchain", False)
        
        if registered_on_blockchain:
            client_id = int(client.cid)
            self.registered_clients[client_id] = client
            return super().register(client)
        else:
            print(f"客戶端 {client.cid} 未在區塊鏈上註冊，拒絕連接")
            return False


# 啟動 Flower 伺服器
def run_server(
    blockchain_connector: BlockchainConnector,
    round_id: int,
    num_rounds: int = 3,
    min_clients: int = 2,
    sample_fraction: float = 1.0
) -> None:
    """啟動 Flower 伺服器，整合區塊鏈功能"""
    
    # 創建客戶端管理器
    client_manager = BlockchainClientManager()
    
    # 創建聯邦學習策略
    strategy = BlockchainFedAvg(
        blockchain_connector=blockchain_connector,
        round_id=round_id,
        fraction_fit=sample_fraction,
        fraction_evaluate=sample_fraction,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        num_rounds=num_rounds
    )
    
    # 啟動伺服器
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        client_manager=client_manager,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="啟動 Flower 聯邦學習伺服器，整合區塊鏈功能")
    parser.add_argument("--contract-address", type=str, required=True, help="智能合約地址")
    parser.add_argument("--round-id", type=int, default=1, help="訓練輪次 ID")
    parser.add_argument("--num-rounds", type=int, default=3, help="每輪訓練迭代次數")
    parser.add_argument("--min-clients", type=int, default=2, help="最小客戶端數量")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="采樣比例")
    parser.add_argument("--node-url", type=str, default="http://127.0.0.1:8545", help="以太坊節點 URL")
    
    args = parser.parse_args()
    
    # 初始化區塊鏈連接器
    blockchain_connector = BlockchainConnector(
        contract_address=args.contract_address,
        node_url=args.node_url
    )
    
    # 啟動伺服器
    run_server(
        blockchain_connector=blockchain_connector,
        round_id=args.round_id,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
        sample_fraction=args.sample_fraction
    )