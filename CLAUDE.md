# CLAUDE.md - Development Guidelines

## Build Commands
- Start local Ethereum node: `npx hardhat node`
- Deploy contracts: `npx hardhat ignition deploy ignition/modules/federated-learning.js --network localhost`
- Run tests: `npx hardhat test`
- Run single test: `npx hardhat test test/Lock.js --grep "Should set the right owner"`
- Lint Solidity: `npx hardhat check`

## Python Commands
- Run server: `python fl/server.py --contract-address <address> --round-id 1`
- Run client: `python fl/client.py --client-id <id> --contract-address <address>`
- Install dependencies: `pip install -r requirements.txt`

## Code Style Guidelines
- **JavaScript**: Use ES6 syntax, 2-space indentation
- **Solidity**: Follow 0.8.20 syntax, 4-space indentation
- **Python**: PEP 8 compliant, type hints required for function parameters and return values
- **Error Handling**: Use try/catch blocks with specific error messages and proper error propagation
- **Imports**: Group and order imports (standard library, third-party, local)
- **Naming**: camelCase for JS/Solidity variables and functions, snake_case for Python
- **Documentation**: Docstrings for all functions, especially public contract functions