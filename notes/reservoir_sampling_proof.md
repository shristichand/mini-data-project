# Mathematical Proof: Uniformity of Reservoir Sampling (Algorithm R)

## 1. Problem Statement
Given a data stream of unknown or large total length $N$, we wish to select a sample of size $n \le N$ such that at any point after $N \ge n$ items have been processed, **every item in the stream has an equal probability $\frac{n}{N}$ of being included in the sample**.

The sampling algorithm must operate in a single pass over the data, requiring memory bounded by $O(n)$ without loading all $N$ items into memory.

---

## 2. Algorithm Description (Algorithm R)

1. **Initialization**: Fill the reservoir $R$ of capacity $n$ with the first $n$ items from the stream (items $1, 2, \dots, n$).
2. **Streaming Step**: For each subsequent item $k$ arriving from the stream ($k = n+1, n+2, \dots, N$):
   - Draw an integer $j$ uniformly at random from the range $[1, k]$.
   - If $j \le n$: Replace the element at position $j$ in the reservoir with item $k$ ($R[j] \leftarrow \text{item } k$).
   - If $j > n$: Discard item $k$ (the reservoir remains unchanged).

---

## 3. Proof of Uniformity by Mathematical Induction

We prove by induction on stream length $N$ that for any $N \ge n$, each item $i \in \{1, 2, \dots, N\}$ resides in the reservoir with probability:
$$P(\text{item } i \in R_N) = \frac{n}{N}$$

### Base Case ($N = n$):
When exactly $n$ items have arrived, all $n$ items are placed into the reservoir.
$$P(\text{item } i \in R_n) = 1 = \frac{n}{n}$$
The base case holds trivially.

---

### Inductive Hypothesis:
Assume that after processing $k$ items (where $k \ge n$), every item $i \in \{1, 2, \dots, k\}$ is in the reservoir with probability:
$$P(\text{item } i \in R_k) = \frac{n}{k}$$

---

### Inductive Step ($k \to k+1$):
Suppose the $(k+1)$-th item arrives. We examine the probability of membership in $R_{k+1}$ for:
1. The new arrival (item $k+1$)
2. Any previously processed item $i \le k$

#### Case 1: Probability that the new arrival (item $k+1$) enters $R_{k+1}$
Item $k+1$ is chosen if the randomly generated index $j \in \{1, 2, \dots, k+1\}$ satisfies $j \le n$.
Because $j$ is drawn uniformly from $k+1$ possible integers:
$$P(\text{item } k+1 \in R_{k+1}) = \frac{n}{k+1}$$
This matches the induction target $\frac{n}{k+1}$.

#### Case 2: Probability that an existing item $i \le k$ remains in $R_{k+1}$
For an existing item $i$ to be in $R_{k+1}$, two independent events must hold:
1. Item $i$ was in the reservoir at step $k$: $E_1 = \{i \in R_k\}$.
2. Item $i$ was **not evicted** at step $k+1$: $E_2 = \{i \text{ not replaced at step } k+1\}$.

From our inductive hypothesis:
$$P(E_1) = \frac{n}{k}$$

Now we calculate the survival probability $P(E_2 \mid E_1)$. Item $i$ survives if either:
- The new item $k+1$ is **not selected** for the reservoir (probability $1 - \frac{n}{k+1}$), OR
- The new item $k+1$ **is selected**, but replaces an item **other than** item $i$. Since there are $n$ positions and the replacement index is chosen uniformly, the probability of replacing one of the other $n-1$ slots is $\frac{n-1}{n}$.

Thus, the conditional survival probability is:
$$P(E_2 \mid E_1) = \left(1 - \frac{n}{k+1}\right) + \left(\frac{n}{k+1}\right) \left(\frac{n-1}{n}\right)$$

Simplifying the algebra:
$$P(E_2 \mid E_1) = \frac{k+1-n}{k+1} + \frac{n-1}{k+1} = \frac{(k+1-n) + (n-1)}{k+1} = \frac{k}{k+1}$$

Finally, applying the chain rule of probability:
$$P(\text{item } i \in R_{k+1}) = P(E_1) \times P(E_2 \mid E_1) = \left(\frac{n}{k}\right) \times \left(\frac{k}{k+1}\right) = \frac{n}{k+1}$$

---

## 4. Conclusion
Both the new arrival and every preceding item share the exact same inclusion probability $\frac{n}{k+1}$.
By the principle of mathematical induction, for any total stream size $N \ge n$:
$$P(\text{item } i \in R_N) = \frac{n}{N} \quad \forall i \in \{1, 2, \dots, N\}$$

Every document in the dataset has an identical, unbiased probability of inclusion, satisfying the definition of a simple random sample without replacement. $\blacksquare$

---

## 5. Memory Boundedness in Batched Execution
In a batched setting over Parquet row chunks:
- Let $B$ be the batch size (e.g., $B = 2,048$).
- As each batch of size $\le B$ is read into memory, we iterate through its rows and update the reservoir of fixed capacity $n$.
- Once the batch is processed, its memory is freed before the next batch is fetched.
- **Maximum concurrent memory footprint**: Exactly $n$ records (the reservoir) $+ B$ records (the active batch).
- For $n = 1,000$ and $B = 2,048$, maximum memory is at most $3,048$ records (~3-6 MB RAM), guaranteeing true $O(n + B)$ scalability even across multi-terabyte datasets.

