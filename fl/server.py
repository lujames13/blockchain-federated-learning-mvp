"""
server.py

Flower 聯邦學習伺服器實作，整合區塊鏈功能 (flwr==1.17)
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
import json

import flwr as fl
import numpy as np
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    NDArrays,
    FitRes,
    EvaluateRes,
    MetricsAggregationFn,
    FitIns
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server import Server

from blockchain_connector import BlockchainConnector


# 自定義聯邦學習策略，整合區塊鏈功能
class BlockchainFedAvg(FedAvg):
    """整合區塊鏈的聯邦學習策略"""

    def __init__(
        self,
        blockchain_connector: BlockchainConnector,
        round_id: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        initial_parameters: Optional[Parameters] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=initial_parameters,
        )
        self.blockchain_connector = blockchain_connector
        self.round_id = round_id
        self.client_updates = {}
        
        # 啟動新輪次
        print(f"開始區塊鏈輪次 {round_id}")
        self.blockchain_connector.start_round(round_id)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """聚合客戶端更新 (flwr 1.17 API)"""
        if not results:
            return None, {}
        
        # 記錄所有參與此輪訓練的客戶端
        for client, fit_res in results:
            client_properties = client.properties
            metrics = fit_res.metrics
            
            if client_properties and "client_id" in client_properties and "model_hash" in metrics:
                client_id = int(client_properties["client_id"])
                model_hash = metrics["model_hash"]
                self.client_updates[client_id] = model_hash
                
                # 接受模型更新
                print(f"接受客戶端 {client_id} 的模型更新，雜湊: {model_hash}")
                self.blockchain_connector.accept_model_update(client_id, self.round_id)
        
        # 使用父類方法進行聚合（flwr 1.17的API參數調整）
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # 如果成功聚合，將結果存儲到區塊鏈
        if aggregated_parameters is not None and server_round > 1:
            parameter_arrays = parameters_to_ndarrays(aggregated_parameters)
            model_hash = self.blockchain_connector.update_global_model(
                self.round_id, parameter_arrays
            )
            print(f"更新輪次 {self.round_id} (伺服器輪次 {server_round}) 的全局模型: {model_hash}")
            
            # 將模型哈希添加到指標中
            metrics["model_hash"] = model_hash
            
        return aggregated_parameters, metrics
    
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
        if server_round == 1:  # 第一輪初始化全局模型
            model_hash = self.blockchain_connector.update_global_model(
                self.round_id, parameter_arrays
            )
            print(f"初始化輪次 {self.round_id} 的全局模型: {model_hash}")
        
        # 呼叫父類方法繼續
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """聚合評估結果"""
        if not results:
            return None, {}
            
        # 使用父類方法進行聚合
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # 最後一輪，處理區塊鏈相關操作
        if server_round == self.num_rounds:
            print(f"完成區塊鏈輪次 {self.round_id}")
            
            # 完成輪次
            self.blockchain_connector.complete_round(self.round_id)
            
            # 獎勵參與者
            for client_id, _ in self.client_updates.items():
                print(f"獎勵客戶端 {client_id}")
                self.blockchain_connector.reward_client(client_id, self.round_id, 100)
                
            metrics["round_completed"] = True
            metrics["clients_rewarded"] = len(self.client_updates)
        
        return aggregated_loss, metrics


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration dict for each round."""
    config = {
        "round": server_round,
        "local_epochs": 1,
        "batch_size": 32,
    }
    return config


def evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """Return evaluation configuration dict for each round."""
    return {"round": server_round}


def main(
    blockchain_connector: BlockchainConnector,
    round_id: int,
    num_rounds: int = 3,
    min_clients: int = 2,
    sample_fraction: float = 1.0,
    host: str = "0.0.0.0",
    port: int = 8080
) -> None:
    """啟動 Flower 伺服器，整合區塊鏈功能"""
    # 創建聯邦學習策略
    strategy = BlockchainFedAvg(
        blockchain_connector=blockchain_connector,
        round_id=round_id,
        fraction_fit=sample_fraction,
        fraction_evaluate=sample_fraction,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )
    
    # 設置服務器
    server = Server(client_manager=fl.server.SimpleClientManager(), strategy=strategy)
    
    # 定義 server_config
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    # 啟動 server
    fl.server.start_server(
        server_address=f"{host}:{port}",
        server=server,
        config=server_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="啟動 Flower 聯邦學習伺服器，整合區塊鏈功能")
    parser.add_argument("--contract-address", type=str, required=True, help="智能合約地址")
    parser.add_argument("--round-id", type=int, default=1, help="訓練輪次 ID")
    parser.add_argument("--num-rounds", type=int, default=3, help="每輪訓練迭代次數")
    parser.add_argument("--min-clients", type=int, default=2, help="最小客戶端數量")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="采樣比例")
    parser.add_argument("--node-url", type=str, default="http://127.0.0.1:8545", help="以太坊節點 URL")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服務器主機地址")
    parser.add_argument("--port", type=int, default=8080, help="服務器端口")
    
    args = parser.parse_args()
    
    # 初始化區塊鏈連接器
    blockchain_connector = BlockchainConnector(
        contract_address=args.contract_address,
        node_url=args.node_url
    )
    
    # 啟動伺服器
    main(
        blockchain_connector=blockchain_connector,
        round_id=args.round_id,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
        sample_fraction=args.sample_fraction,
        host=args.host,
        port=args.port
    )