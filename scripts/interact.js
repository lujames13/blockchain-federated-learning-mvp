// Import required modules
const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Contract interaction functions
async function getContractInstance() {
  const [deployer] = await ethers.getSigners();
  console.log("Interacting with contracts using account:", deployer.address);

  // Find the most recent deployment from ignition
  const deploymentsDir = path.join(__dirname, "../ignition/deployments");
  const deploymentFolders = fs.readdirSync(deploymentsDir);
  const latestDeployment = deploymentFolders.sort().pop();
  
  if (!latestDeployment) {
    throw new Error("No deployments found. Deploy the contract first.");
  }

  // Load artifacts from the latest deployment
  const artifactPath = path.join(deploymentsDir, latestDeployment, "artifacts.json");
  const artifacts = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
  
  // Get the contract address
  const contractAddress = artifacts.contracts.FederatedLearning.address;
  console.log("FederatedLearning contract address:", contractAddress);

  // Get the contract factory
  const FederatedLearning = await ethers.getContractFactory("FederatedLearning");
  
  // Create a contract instance
  return FederatedLearning.attach(contractAddress);
}

// Function to get system status
async function getStatus() {
  const contract = await getContractInstance();
  const status = await contract.getSystemStatus();
  
  console.log("System Status:");
  console.log("  Total Clients:", status.totalClients.toString());
  console.log("  Total Rounds:", status.totalRounds.toString());
  console.log("  Current Round:", status.currentRound.toString());
  console.log("  Current Round Status:", status.currentRoundStatus);
  
  return status;
}

// Function to register a client
async function registerClient(clientId) {
  const contract = await getContractInstance();
  
  console.log(`Registering client with ID: ${clientId}`);
  const tx = await contract.registerClient(clientId);
  await tx.wait();
  
  console.log(`Client ${clientId} registered successfully!`);
  
  // Get and print client info
  const clientInfo = await contract.getClientInfo(clientId);
  console.log("Client Info:");
  console.log("  Address:", clientInfo.clientAddress);
  console.log("  Status:", clientInfo.status);
  console.log("  Contribution Score:", clientInfo.contributionScore.toString());
  console.log("  Last Update Timestamp:", new Date(clientInfo.lastUpdateTimestamp * 1000).toISOString());
  
  return clientInfo;
}

// Function to start a new round
async function startRound(roundId) {
  const contract = await getContractInstance();
  
  console.log(`Starting round with ID: ${roundId}`);
  const tx = await contract.startRound(roundId);
  await tx.wait();
  
  console.log(`Round ${roundId} started successfully!`);
  
  // Get and print round info
  const roundInfo = await contract.getRoundInfo(roundId);
  console.log("Round Info:");
  console.log("  ID:", roundInfo.id.toString());
  console.log("  Status:", roundInfo.status);
  console.log("  Start Time:", new Date(roundInfo.startTime * 1000).toISOString());
  
  return roundInfo;
}

// Function to submit a model update
async function submitModelUpdate(clientId, roundId, modelUpdateHash) {
  const contract = await getContractInstance();
  
  console.log(`Submitting model update for client ${clientId} in round ${roundId}`);
  const tx = await contract.submitModelUpdate(clientId, roundId, modelUpdateHash);
  await tx.wait();
  
  console.log("Model update submitted successfully!");
  return true;
}

// Function to accept a model update
async function acceptModelUpdate(clientId, roundId) {
  const contract = await getContractInstance();
  
  console.log(`Accepting model update for client ${clientId} in round ${roundId}`);
  const tx = await contract.acceptModelUpdate(clientId, roundId);
  await tx.wait();
  
  console.log("Model update accepted successfully!");
  return true;
}

// Function to update the global model
async function updateGlobalModel(roundId, globalModelHash) {
  const contract = await getContractInstance();
  
  console.log(`Updating global model for round ${roundId}`);
  const tx = await contract.updateGlobalModel(roundId, globalModelHash);
  await tx.wait();
  
  console.log("Global model updated successfully!");
  return true;
}

// Function to complete a round
async function completeRound(roundId) {
  const contract = await getContractInstance();
  
  console.log(`Completing round ${roundId}`);
  const tx = await contract.completeRound(roundId);
  await tx.wait();
  
  console.log(`Round ${roundId} completed successfully!`);
  
  // Get and print round info
  const roundInfo = await contract.getRoundInfo(roundId);
  console.log("Round Info:");
  console.log("  ID:", roundInfo.id.toString());
  console.log("  Status:", roundInfo.status);
  console.log("  Start Time:", new Date(roundInfo.startTime * 1000).toISOString());
  console.log("  End Time:", new Date(roundInfo.endTime * 1000).toISOString());
  console.log("  Participant Count:", roundInfo.participantCount.toString());
  console.log("  Completed Updates:", roundInfo.completedUpdates.toString());
  console.log("  Global Model Hash:", roundInfo.globalModelHash);
  
  return roundInfo;
}

// Function to reward a client
async function rewardClient(clientId, roundId, rewardAmount) {
  const contract = await getContractInstance();
  
  console.log(`Rewarding client ${clientId} for round ${roundId} with ${rewardAmount} tokens`);
  const tx = await contract.rewardClient(clientId, roundId, rewardAmount);
  await tx.wait();
  
  console.log("Client rewarded successfully!");
  return true;
}

// Parse command line arguments
async function main() {
  const args = process.argv.slice(2);
  const functionName = args.find(arg => arg.startsWith("--func="))?.split("=")[1];
  
  if (!functionName) {
    console.log("Please provide a function name with --func=<functionName>");
    return;
  }
  
  switch (functionName) {
    case "getStatus":
      await getStatus();
      break;
    case "registerClient":
      const clientId = args.find(arg => arg.startsWith("--clientId="))?.split("=")[1];
      if (!clientId) {
        console.log("Please provide a client ID with --clientId=<id>");
        return;
      }
      await registerClient(clientId);
      break;
    case "startRound":
      const roundId = args.find(arg => arg.startsWith("--roundId="))?.split("=")[1];
      if (!roundId) {
        console.log("Please provide a round ID with --roundId=<id>");
        return;
      }
      await startRound(roundId);
      break;
    case "submitModelUpdate":
      const submitClientId = args.find(arg => arg.startsWith("--clientId="))?.split("=")[1];
      const submitRoundId = args.find(arg => arg.startsWith("--roundId="))?.split("=")[1];
      const modelHash = args.find(arg => arg.startsWith("--hash="))?.split("=")[1];
      if (!submitClientId || !submitRoundId || !modelHash) {
        console.log("Please provide --clientId=<id> --roundId=<id> --hash=<modelHash>");
        return;
      }
      await submitModelUpdate(submitClientId, submitRoundId, modelHash);
      break;
    case "acceptModelUpdate":
      const acceptClientId = args.find(arg => arg.startsWith("--clientId="))?.split("=")[1];
      const acceptRoundId = args.find(arg => arg.startsWith("--roundId="))?.split("=")[1];
      if (!acceptClientId || !acceptRoundId) {
        console.log("Please provide --clientId=<id> --roundId=<id>");
        return;
      }
      await acceptModelUpdate(acceptClientId, acceptRoundId);
      break;
    case "updateGlobalModel":
      const updateRoundId = args.find(arg => arg.startsWith("--roundId="))?.split("=")[1];
      const globalHash = args.find(arg => arg.startsWith("--hash="))?.split("=")[1];
      if (!updateRoundId || !globalHash) {
        console.log("Please provide --roundId=<id> --hash=<globalModelHash>");
        return;
      }
      await updateGlobalModel(updateRoundId, globalHash);
      break;
    case "completeRound":
      const completeRoundId = args.find(arg => arg.startsWith("--roundId="))?.split("=")[1];
      if (!completeRoundId) {
        console.log("Please provide a round ID with --roundId=<id>");
        return;
      }
      await completeRound(completeRoundId);
      break;
    case "rewardClient":
      const rewardClientId = args.find(arg => arg.startsWith("--clientId="))?.split("=")[1];
      const rewardRoundId = args.find(arg => arg.startsWith("--roundId="))?.split("=")[1];
      const amount = args.find(arg => arg.startsWith("--amount="))?.split("=")[1];
      if (!rewardClientId || !rewardRoundId || !amount) {
        console.log("Please provide --clientId=<id> --roundId=<id> --amount=<rewardAmount>");
        return;
      }
      await rewardClient(rewardClientId, rewardRoundId, amount);
      break;
    default:
      console.log(`Unknown function: ${functionName}`);
      console.log("Available functions: getStatus, registerClient, startRound, submitModelUpdate, acceptModelUpdate, updateGlobalModel, completeRound, rewardClient");
  }
}

// Execute the script
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

// Export functions for programmatic use
module.exports = {
  getContractInstance,
  getStatus,
  registerClient,
  startRound,
  submitModelUpdate,
  acceptModelUpdate,
  updateGlobalModel,
  completeRound,
  rewardClient
};