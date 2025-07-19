# TODO: Refactor `fluent-ai/packages/memory` Crate

## Milestone 1: Refactor the `cognitive` module

The `cognitive` module is the largest and most complex, and it will be broken down first.

### Task 1.1: Decompose `cognitive/committee_old.rs` (1010 lines)

- **Objective**: This file is marked as "old" and is a prime candidate for removal or refactoring.
- **Architecture**: I will first analyze the file to determine if it's still in use. If it is, I will break it down into smaller, more focused modules. If not, I will remove it entirely.
- **Implementation**:
    - **Step 1**: Use `grep` to search for usages of `committee_old` throughout the codebase.
    - **Step 2**: If no usages are found, delete the file.
    - **Step 3**: If usages are found, analyze the contents and create new modules such as `cognitive/committee/consensus.rs`, `cognitive/committee/evaluation.rs`, and `cognitive/committee/membership.rs`.
    - **Step 4**: Move the relevant code from `committee_old.rs` to the new modules.
    - **Step 5**: Update all references to `committee_old` to use the new modules.
    - **Step 6**: Delete `committee_old.rs`.
- **QA**: Verify that all tests pass and the application runs correctly after the refactoring. Ensure no functionality has been lost.

### Task 1.2: Decompose `cognitive/quantum/ml_decoder.rs` (979 lines)

- **Objective**: This file is too large and needs to be broken down into smaller, more manageable modules.
- **Architecture**: The file will be split into logical components such as `decoder`, `training`, and `utility` functions.
- **Implementation**:
    - **Step 1**: Create a new directory `cognitive/quantum/ml_decoder`.
    - **Step 2**: Create `cognitive/quantum/ml_decoder/decoder.rs` for the core decoding logic.
    - **Step 3**: Create `cognitive/quantum/ml_decoder/training.rs` for the machine learning training logic.
    - **Step 4**: Create `cognitive/quantum/ml_decoder/utils.rs` for helper functions.
    - **Step 5**: Move the relevant code from the original `ml_decoder.rs` to the new files.
    - **Step 6**: Create `cognitive/quantum/ml_decoder/mod.rs` to expose the public API.
- **QA**: Ensure all unit tests for the `ml_decoder` pass and that the public interface remains unchanged.

### Task 1.3: Decompose `cognitive/quantum_mcts.rs` (757 lines)

- **Objective**: Refactor the Monte Carlo Tree Search implementation for quantum systems.
- **Architecture**: The file will be broken down into `node`, `search`, and `policy` modules.
- **Implementation**:
    - **Step 1**: Create a new directory `cognitive/quantum_mcts`.
    - **Step 2**: Create `cognitive/quantum_mcts/node.rs` for the tree node implementation.
    - **Step 3**: Create `cognitive/quantum_mcts/search.rs` for the MCTS algorithm.
    - **Step 4**: Create `cognitive/quantum_mcts/policy.rs` for the selection and expansion policies.
    - **Step 5**: Move the code into the new modules and update the `mod.rs` file.
- **QA**: Verify that the MCTS algorithm behaves as expected through integration tests.

## Milestone 2: Refactor the `memory` module

The `memory` module is the heart of this package and requires careful refactoring.

### Task 2.1: Decompose `memory/memory_type.rs` (761 lines)

- **Objective**: This file defines various memory types and is too large.
- **Architecture**: Each memory type will be moved into its own file under a new `memory/types` directory.
- **Implementation**:
    - **Step 1**: Create a `memory/types` directory.
    - **Step 2**: For each memory type (e.g., `Episodic`, `Semantic`, `Procedural`), create a new file in `memory/types`.
    - **Step 3**: Move the corresponding code into the new files.
    - **Step 4**: Create a `memory/types/mod.rs` to export all the types.
- **QA**: Ensure that all memory types are correctly exported and that all existing tests pass.

### Task 2.2: Decompose `memory/memory_manager.rs` (636 lines)

- **Objective**: The `MemoryManager` is doing too much and needs to be broken down.
- **Architecture**: The `MemoryManager` will be split into smaller, more focused managers for different aspects of memory management, such as `storage`, `retrieval`, and `evolution`.
- **Implementation**:
    - **Step 1**: Create `memory/storage_manager.rs`.
    - **Step 2**: Create `memory/retrieval_manager.rs`.
    - **Step 3**: Create `memory/evolution_manager.rs`.
    - **Step 4**: Move the relevant logic from `memory_manager.rs` to the new manager files.
    - **Step 5**: Refactor `memory_manager.rs` to be a facade that delegates to the new managers.
- **QA**: All tests related to memory management must pass. The public API of `MemoryManager` should remain backward compatible if possible.

## Milestone 3: Refactor Remaining Modules

This milestone will address the remaining files that exceed the 300-line limit.

### Task 3.1: Decompose `graph/entity.rs` (705 lines)

- **Objective**: The `entity.rs` file is too large and needs to be broken down.
- **Architecture**: The file will be split into smaller modules based on the entity type or functionality.
- **Implementation**:
    - **Step 1**: Create a `graph/entity` directory.
    - **Step 2**: Create files for different entity-related logic, such as `graph/entity/node.rs`, `graph/entity/relationship.rs`, and `graph/entity/properties.rs`.
    - **Step 3**: Move the code into the new files and update the `mod.rs`.
- **QA**: Ensure that all graph-related tests pass and that the entity API is consistent.