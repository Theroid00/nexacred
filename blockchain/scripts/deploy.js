const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🚀 Deploying NexaCred smart contracts...");

  // Get the contract factories
  const CreditScore = await hre.ethers.getContractFactory("CreditScore");
  const NexaCred = await hre.ethers.getContractFactory("NexaCred");

  // Deploy Credit Score contract first
  console.log("\n📊 Deploying CreditScore contract...");
  const creditScore = await CreditScore.deploy();
  await creditScore.deployed();
  console.log(`✅ CreditScore deployed to: ${creditScore.address}`);

  // Deploy NexaCred lending contract
  console.log("\n💰 Deploying NexaCred lending contract...");
  const nexaCred = await NexaCred.deploy();
  await nexaCred.deployed();
  console.log(`✅ NexaCred deployed to: ${nexaCred.address}`);

  // Setup initial configuration
  console.log("\n⚙️ Setting up initial configuration...");
  
  // Get deployer account
  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deployer account: ${deployer.address}`);
  
  // Authorize NexaCred contract to update credit scores
  const authTx = await creditScore.setAuthorizedUpdater(nexaCred.address, true);
  await authTx.wait();
  console.log("✅ NexaCred contract authorized to update credit scores");

  // Create deployment info for backend integration
  const deploymentInfo = {
    network: hre.network.name,
    chainId: hre.network.config.chainId,
    contracts: {
      CreditScore: {
        address: creditScore.address,
        deployer: deployer.address,
        blockNumber: creditScore.deployTransaction.blockNumber
      },
      NexaCred: {
        address: nexaCred.address,
        deployer: deployer.address,
        blockNumber: nexaCred.deployTransaction.blockNumber
      }
    },
    deployedAt: new Date().toISOString(),
    deployer: deployer.address
  };

  // Save deployment info for backend use
  const deploymentPath = path.join(__dirname, "..", "deployment.json");
  fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`📄 Deployment info saved to: ${deploymentPath}`);

  // Generate environment variables for backend
  const envVars = `
# Blockchain Configuration (Generated from deployment)
BLOCKCHAIN_NETWORK=${hre.network.name}
CREDIT_SCORE_CONTRACT_ADDRESS=${creditScore.address}
NEXACRED_CONTRACT_ADDRESS=${nexaCred.address}
WEB3_PROVIDER_URL=${hre.network.config.url}
`;

  const envPath = path.join(__dirname, "..", ".env.blockchain");
  fs.writeFileSync(envPath, envVars.trim());
  console.log(`🔧 Environment variables saved to: ${envPath}`);

  console.log("\n🎉 Deployment completed successfully!");
  console.log("\n📋 Next steps:");
  console.log("1. Copy the contract addresses to your backend configuration");
  console.log("2. Update web3_integration.py with the deployed contract addresses");
  console.log("3. Test the integration with the Flask backend");
  
  console.log("\n📊 Deployment Summary:");
  console.log(`Network: ${hre.network.name}`);
  console.log(`CreditScore: ${creditScore.address}`);
  console.log(`NexaCred: ${nexaCred.address}`);
  console.log(`Deployer: ${deployer.address}`);
}

// Handle deployment errors
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:");
    console.error(error);
    process.exit(1);
  });
