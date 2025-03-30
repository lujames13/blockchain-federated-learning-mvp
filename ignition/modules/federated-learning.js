import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

/**
 * @dev Ignition module for deploying and initializing the FederatedLearning contract
 */
export default buildModule("FederatedLearningModule", (m) => {
  // Deploy FederatedLearning contract
  const federatedLearning = m.contract("FederatedLearning");
  
  // Initialize the contract
  m.call(federatedLearning, "initialize", []);
  
  // Return the deployed contract
  return { federatedLearning };
});