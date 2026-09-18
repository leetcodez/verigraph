# Distributed Systems: Consensus Protocols & Fault Tolerance

## 1. Introduction to Distributed Consensus
Distributed consensus is the fundamental problem of getting multiple nodes in an asynchronous network to agree on a sequence of state machine operations despite node failures, message drops, and network delays. The two primary families of algorithms that solve this are Paxos and Raft.

## 2. The Raft Consensus Algorithm
Raft was designed by Ongaro and Ousterhout to provide the same safety guarantees as Multi-Paxos while significantly improving understandability. Raft decomposes consensus into three independent subproblems: leader election, log replication, and safety.

### 2.1 Leader Election and Quorum
At any given moment, each Raft node is in one of three states: Leader, Follower, or Candidate. Nodes transition from Follower to Candidate if they do not receive a heartbeat within a randomized election timeout (typically 150ms - 300ms).
- A candidate requests votes from other peers.
- A candidate wins the election if it receives votes from a Majority Quorum of nodes (at least `(N/2) + 1` in an N-node cluster).
- By requiring a Majority Quorum, Raft strictly prevents Split-Brain scenarios during network partitions. If a 5-node cluster is partitioned into 3 and 2 nodes, only the partition with 3 nodes can elect a Leader or commit logs.

### 2.2 Log Replication & Heartbeats
Once elected, the Leader accepts client requests, appends log entries, and replicates them to Follower nodes via AppendEntries RPCs. 
- The Leader periodically broadcasts heartbeat messages to Followers to maintain its authority and reset follower election timers.
- An entry is considered committed once it has been replicated to a majority of nodes.
- Followers apply committed entries to their local state machines in strict log order.

## 3. Paxos and Multi-Paxos
Paxos, designed by Leslie Lamport, provides consensus across two phases: Prepare-Promise (Phase 1) and Propose-Accept (Phase 2). While theoretically elegant, classic Paxos is notoriously difficult to implement in production because it only decides a single value. Multi-Paxos extends this by amortizing Phase 1 across a stream of log entries, but introduces subtle edge cases during leader changes.

## 4. Split-Brain Phenomenon & Mitigation
Split-Brain occurs when a distributed cluster is severed by a network partition, causing disjoint subsets of nodes to simultaneously believe they possess leadership authority. 
- In systems without strict quorum guarantees, two leaders may accept conflicting writes, permanently corrupting persistent data.
- Raft mitigates Split-Brain by enforcing that any log commitment or leader election must contact a strict majority of total cluster nodes. In an asymmetric partition, the minority partition cannot make progress, ensuring Linearizability.
