"""Batch 13 (synth_61 .. synth_65): Strings & Tries II — Substring Palindrome Queries, XOR MST, KMP Border Tree LCA, Suffix Array Substring Frequency Ranges, Multi-Pattern Aho-Corasick Censoring."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_13() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_61: PALINDROME RADIUS (2100) - Manacher + RMQ + Binary Search
    # ------------------------------------------------------------------
    def gen_61(n: int, q: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        if kind == "all_a":
            chars = ["a"] * n
            for _ in range(max(1, n // 5000)):
                chars[r.randrange(n)] = "b"
            s = "".join(chars)
        elif kind == "periodic":
            pat = "abacaba"
            s = (pat * (n // len(pat) + 1))[:n]
        else:
            # Planted palindromes inside small-alphabet text
            chars = [r.choice("abc") for _ in range(n)]
            for _ in range(max(5, n // 40)):
                c = r.randrange(n)
                rad = r.randint(1, min(80, c, n - 1 - c) if min(c, n - 1 - c) >= 1 else 0)
                for d in range(1, rad + 1):
                    chars[c + d] = chars[c - d]
            s = "".join(chars)
        lines = [f"{n} {q}", s]
        for _ in range(q):
            if kind == "all_a" and r.random() < 0.8:
                l = r.randint(1, max(1, n // 5))
                rr = r.randint(max(l, 4 * n // 5), n)
            else:
                l, rr = r.randint(1, n), r.randint(1, n)
                if l > rr:
                    l, rr = rr, l
            lines.append(f"{l} {rr}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_61",
        title="Palindrome Radius",
        rating=2100,
        tags=["strings", "manacher", "sparse table", "binary search"],
        notes="Precompute odd and even Manacher radii d1, d2 with O(1) Sparse Table RMQ; for each query [L, R], binary search the largest feasible odd and even radius.",
        story_and_spec="""You are given a lowercase string S of length N (1-indexed) and Q queries. Each query gives two indices L and R (1 <= L <= R <= N) and asks for the maximum length of a palindromic contiguous substring S[a..b] satisfying L <= a <= b <= R.

Input
The first line contains N and Q.
The second line contains the string S of length N.
Each of the next Q lines contains L and R.

Output
For each query print the length of the longest palindrome inside S[L..R] on its own line.

Constraints
1 <= N, Q <= 150000
S consists of lowercase English letters ('a'-'z')
1 <= L <= R <= N""",
        samples=[
            "9 4\nbabadabba\n1 5\n4 9\n5 8\n2 3\n",
        ],
        small_inputs=[
            "1 1\nz\n1 1\n",
            gen_61(180, 250, 6101),
            gen_61(350, 350, 6102, "all_a"),
            gen_61(300, 300, 6103, "periodic"),
        ],
        large_inputs=[
            gen_61(150000, 150000, 6111, "all_a"),
            gen_61(150000, 150000, 6112, "periodic"),
            gen_61(150000, 150000, 6113, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct SparseMax {
    int n, lg;
    vector<vector<int>> st;
    void build(const vector<int>& a) {
        n = (int)a.size();
        if (n == 0) return;
        lg = __lg(n) + 1;
        st.assign(lg, vector<int>(n));
        st[0] = a;
        for (int k = 1; k < lg; k++) {
            int h = 1 << (k - 1);
            for (int i = 0; i + (1 << k) <= n; i++)
                st[k][i] = max(st[k - 1][i], st[k - 1][i + h]);
        }
    }
    inline int query(int l, int r) const {
        int k = __lg(r - l + 1);
        return max(st[k][l], st[k][r - (1 << k) + 1]);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;

    // d1[i]: max odd radius r >= 1 s.t. s[i-r+1 .. i+r-1] is palindrome
    vector<int> d1(n);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
        while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) k++;
        d1[i] = k--;
        if (i + k > r) { l = i - k; r = i + k; }
    }

    // d2[i] (0 <= i < n-1): max even radius r >= 0 s.t. s[i-r+1 .. i+r] is palindrome
    vector<int> d2(max(0, n - 1), 0);
    for (int i = 0, l = 0, r = -1; i < n - 1; i++) {
        int k = (i + 1 > r) ? 0 : min(d2[l + r - i - 1], r - i);
        while (0 <= i - k && i + 1 + k < n && s[i - k] == s[i + 1 + k]) k++;
        d2[i] = k;
        if (i + k > r) { l = i - k + 1; r = i + k; }
    }

    SparseMax rmq1, rmq2;
    rmq1.build(d1);
    if (n > 1) rmq2.build(d2);

    while (q--) {
        int l, r;
        cin >> l >> r;
        --l; --r;
        int len = r - l + 1;

        // Odd radius k in [1, (len + 1) / 2]
        int lo = 1, hi = (len + 1) / 2;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (rmq1.query(l + mid - 1, r - mid + 1) >= mid) lo = mid;
            else hi = mid - 1;
        }
        int ans = 2 * lo - 1;

        // Even radius k in [1, len / 2]
        if (len >= 2) {
            int elo = 0, ehi = len / 2;
            while (elo < ehi) {
                int mid = (elo + ehi + 1) >> 1;
                if (rmq2.query(l + mid - 1, r - mid) >= mid) elo = mid;
                else ehi = mid - 1;
            }
            ans = max(ans, 2 * elo);
        }
        cout << ans << '\n';
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;
    while (q--) {
        int l, r;
        cin >> l >> r;
        --l; --r;
        int best = 1;
        for (int c = l; c <= r; c++) {
            int k = 1;
            while (c - k >= l && c + k <= r && s[c - k] == s[c + k]) k++;
            best = max(best, 2 * k - 1);
            int e = 0;
            while (c - e >= l && c + 1 + e <= r && s[c - e] == s[c + 1 + e]) e++;
            best = max(best, 2 * e);
        }
        cout << best << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_62: XOR HIGHWAY (2300) - Minimum Spanning Tree Under Bitwise XOR
    # ------------------------------------------------------------------
    def gen_62(n: int, bits: int, seed: int) -> str:
        r = random.Random(seed)
        hi = (1 << bits) - 1
        a = [r.randint(0, hi) for _ in range(n)]
        return f"{n}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_62",
        title="XOR Highway",
        rating=2300,
        tags=["trees", "dsu", "bitmasks", "tries", "divide and conquer"],
        notes="CF 888G Xor-MST: sort unique values, split recursively by highest differing bit b, connect the 0-half and 1-half via minimum-XOR binary trie query in O(N log^2 V).",
        story_and_spec="""There are N data centers numbered 1 to N, where data center i has routing key A_i. You can build a bidirectional fiber link between ANY pair of data centers (i, j) at a cost equal to the bitwise XOR of their keys, A_i XOR A_j.

Find the minimum total cost required to connect all N data centers into a single connected network.

Input
The first line contains N.
The second line contains A_1 ... A_N.

Output
Print a single integer: the minimum spanning tree cost.

Constraints
1 <= N <= 160000
0 <= A_i < 2^30""",
        samples=[
            "5\n1 2 3 4 5\n",
            "4\n7 7 10 10\n",
        ],
        small_inputs=[
            "1\n12345\n",
            gen_62(400, 12, 6201),
            gen_62(1800, 30, 6202),
        ],
        large_inputs=[
            gen_62(160000, 30, 6211),
            gen_62(160000, 20, 6212),
            gen_62(160000, 30, 6213),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct BinaryTrie {
    vector<array<int, 2>> ch;
    int sz = 0;
    void init(int max_nodes) {
        ch.resize(max_nodes);
        reset();
    }
    void reset() {
        ch[0] = {-1, -1};
        sz = 1;
    }
    void insert(int x, int top_bit) {
        int u = 0;
        for (int b = top_bit; b >= 0; b--) {
            int bit = (x >> b) & 1;
            if (ch[u][bit] == -1) {
                ch[sz] = {-1, -1};
                ch[u][bit] = sz++;
            }
            u = ch[u][bit];
        }
    }
    int min_xor(int x, int top_bit) const {
        int u = 0, res = 0;
        for (int b = top_bit; b >= 0; b--) {
            int bit = (x >> b) & 1;
            if (ch[u][bit] != -1) {
                u = ch[u][bit];
            } else {
                res |= (1 << b);
                u = ch[u][bit ^ 1];
            }
        }
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    sort(a.begin(), a.end());
    a.erase(unique(a.begin(), a.end()), a.end());
    int m = (int)a.size();
    if (m <= 1) {
        cout << 0 << '\n';
        return 0;
    }

    BinaryTrie trie;
    trie.init(m * 30 + 5);

    function<long long(int, int, int)> solve = [&](int l, int r, int bit) -> long long {
        if (bit < 0 || l >= r) return 0;
        int mid = l;
        while (mid <= r && !((a[mid] >> bit) & 1)) mid++;
        long long ans = solve(l, mid - 1, bit - 1) + solve(mid, r, bit - 1);
        if (l < mid && mid <= r) {
            trie.reset();
            int L1 = l, R1 = mid - 1, L2 = mid, R2 = r;
            if (R1 - L1 > R2 - L2) { swap(L1, L2); swap(R1, R2); }
            for (int i = L1; i <= R1; i++) trie.insert(a[i], bit - 1);
            int best = INT_MAX;
            for (int i = L2; i <= R2; i++) best = min(best, trie.min_xor(a[i], bit - 1));
            ans += (1LL << bit) + best;
        }
        return ans;
    };

    cout << solve(0, m - 1, 29) << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    vector<int> best(n, INT_MAX);
    vector<char> used(n, 0);
    best[0] = 0;
    long long total = 0;
    for (int it = 0; it < n; it++) {
        int u = -1;
        for (int i = 0; i < n; i++) {
            if (!used[i] && (u == -1 || best[i] < best[u])) u = i;
        }
        used[u] = 1;
        total += best[u];
        int au = a[u];
        for (int v = 0; v < n; v++) if (!used[v]) {
            int w = au ^ a[v];
            if (w < best[v]) best[v] = w;
        }
    }
    cout << total << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_63: COMMON BORDERS (1900) - KMP Failure Tree + LCA
    # ------------------------------------------------------------------
    def gen_63(n: int, q: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        if kind == "all_a":
            s = "a" * n
        elif kind == "thue_fib":
            a, b = "a", "ab"
            while len(b) < n:
                a, b = b, b + a
            s = b[:n]
        else:
            pat = "".join(r.choice("ab") for _ in range(12))
            chars = list((pat * (n // 12 + 1))[:n])
            for _ in range(n // 25):
                chars[r.randrange(n)] = r.choice("ab")
            s = "".join(chars)
        lines = [f"{n} {q}", s]
        for _ in range(q):
            if kind == "all_a" and r.random() < 0.85:
                p = r.randint(max(1, 4 * n // 5), n)
                qq = r.randint(1, max(1, n // 5))
                if r.random() < 0.5:
                    p, qq = qq, p
            else:
                p, qq = r.randint(1, n), r.randint(1, n)
            lines.append(f"{p} {qq}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_63",
        title="Common Borders",
        rating=1900,
        tags=["strings", "kmp", "trees", "lca"],
        notes="Build the KMP border tree on 0..N where parent(i) = pi[i]; proper borders of S[1..p] are ancestors of pi[p], so the answer is LCA(pi[p], pi[q]).",
        story_and_spec="""A proper border of a non-empty string T is a string B with 0 <= |B| < |T| that occurs BOTH as a prefix of T and as a suffix of T (the empty string of length 0 is always a proper border).

You are given a string S of length N (1-indexed) and Q queries. Each query specifies two prefix lengths P and R (1 <= P, R <= N): find the maximum length of a string that is simultaneously a proper border of S[1..P] AND a proper border of S[1..R].

Input
The first line contains N and Q.
The second line contains S.
Each of the next Q lines contains P and R.

Output
For each query print the maximum common proper border length on its own line.

Constraints
1 <= N, Q <= 200000
S consists of lowercase English letters ('a'-'z')
1 <= P, R <= N""",
        samples=[
            "7 4\nabababa\n3 5\n5 5\n1 7\n4 6\n",
        ],
        small_inputs=[
            "1 1\na\n1 1\n",
            gen_63(400, 500, 6301, "thue_fib"),
            gen_63(1200, 1200, 6302, "all_a"),
        ],
        large_inputs=[
            gen_63(200000, 200000, 6311, "all_a"),
            gen_63(200000, 200000, 6312, "thue_fib"),
            gen_63(200000, 200000, 6313, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;

    vector<int> pi(n + 1, 0), dep(n + 1, 0);
    int LOG = __lg(n + 1) + 2;
    vector<vector<int>> up(LOG, vector<int>(n + 1, 0));

    for (int i = 2; i <= n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[j] != s[i - 1]) j = pi[j];
        if (s[j] == s[i - 1]) j++;
        pi[i] = j;
    }
    for (int i = 1; i <= n; i++) {
        dep[i] = dep[pi[i]] + 1;
        up[0][i] = pi[i];
        for (int b = 1; b < LOG; b++) up[b][i] = up[b - 1][up[b - 1][i]];
    }

    auto lca = [&](int u, int v) {
        if (dep[u] < dep[v]) swap(u, v);
        int d = dep[u] - dep[v];
        for (int b = 0; b < LOG; b++) if ((d >> b) & 1) u = up[b][u];
        if (u == v) return u;
        for (int b = LOG - 1; b >= 0; b--)
            if (up[b][u] != up[b][v]) { u = up[b][u]; v = up[b][v]; }
        return up[0][u];
    };

    while (q--) {
        int p, r;
        cin >> p >> r;
        cout << lca(pi[p], pi[r]) << '\n';
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;
    vector<int> pi(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[j] != s[i - 1]) j = pi[j];
        if (s[j] == s[i - 1]) j++;
        pi[i] = j;
    }
    while (q--) {
        int a, b;
        cin >> a >> b;
        a = pi[a];
        b = pi[b];
        while (a != b) {
            if (a > b) a = pi[a];
            else b = pi[b];
        }
        cout << a << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_64: SUBSTRING FREQUENCIES (2100) - Suffix Array + LCP Monotonic Stack
    # ------------------------------------------------------------------
    def gen_64(n: int, q: int, alpha: int, seed: int) -> str:
        r = random.Random(seed)
        letters = "abcdefghijklmnopqrstuvwxyz"[:alpha]
        s = "".join(r.choice(letters) for _ in range(n))
        lines = [f"{n} {q}", s]
        for _ in range(q):
            a = r.randint(1, n if r.random() < 0.3 else max(1, min(n, 50)))
            b = r.randint(a, n)
            lines.append(f"{a} {b}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_64",
        title="Substring Frequencies",
        rating=2100,
        tags=["strings", "suffix array", "monotonic stack", "combinatorics"],
        notes="Build Suffix Array + Kasai LCP; decompose LCP via Cartesian-tree / monotonic stack intervals (plus freq=1 suffixes) to tabulate exact distinct-substring counts for every occurrence multiplicity 1..N.",
        story_and_spec="""You are given a string S of length N and Q queries. Each query specifies two integers A and B (1 <= A <= B <= N): count how many DISTINCT non-empty substrings T of S occur in S at least A times and at most B times (overlapping occurrences count; for example, "aba" occurs twice in "ababa").

Input
The first line contains N and Q.
The second line contains S.
Each of the next Q lines contains A and B.

Output
For each query print the number of distinct substrings whose occurrence count lies in [A, B] on its own line.

Constraints
1 <= N, Q <= 160000
S consists of lowercase English letters ('a'-'z')
1 <= A <= B <= N""",
        samples=[
            "5 4\nababa\n1 5\n2 2\n3 5\n1 1\n",
        ],
        small_inputs=[
            "4 3\naaaa\n1 4\n2 3\n4 4\n",
            gen_64(250, 200, 3, 6401),
            gen_64(600, 300, 2, 6402),
        ],
        large_inputs=[
            gen_64(150000, 150000, 3, 6411),
            gen_64(160000, 160000, 2, 6412),
            gen_64(160000, 160000, 10, 6413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;

    // O(N log N) radix / counting Suffix Array
    vector<int> sa(n), rk(n), nrk(n), cnt(max(256, n) + 1, 0), tmp(n);
    for (int i = 0; i < n; i++) cnt[(unsigned char)s[i]]++;
    for (int i = 1; i < 256; i++) cnt[i] += cnt[i - 1];
    for (int i = n - 1; i >= 0; i--) sa[--cnt[(unsigned char)s[i]]] = i;
    rk[sa[0]] = 0;
    int classes = 1;
    for (int i = 1; i < n; i++) {
        if (s[sa[i]] != s[sa[i - 1]]) classes++;
        rk[sa[i]] = classes - 1;
    }
    for (int h = 1; h < n && classes < n; h <<= 1) {
        int p = 0;
        for (int i = n - h; i < n; i++) tmp[p++] = i;
        for (int i = 0; i < n; i++) if (sa[i] >= h) tmp[p++] = sa[i] - h;
        fill(cnt.begin(), cnt.begin() + classes, 0);
        for (int i = 0; i < n; i++) cnt[rk[tmp[i]]]++;
        for (int i = 1; i < classes; i++) cnt[i] += cnt[i - 1];
        for (int i = n - 1; i >= 0; i--) sa[--cnt[rk[tmp[i]]]] = tmp[i];
        nrk[sa[0]] = 0;
        classes = 1;
        for (int i = 1; i < n; i++) {
            int u = sa[i], v = sa[i - 1];
            int ru2 = (u + h < n ? rk[u + h] : -1);
            int rv2 = (v + h < n ? rk[v + h] : -1);
            if (rk[u] != rk[v] || ru2 != rv2) classes++;
            nrk[u] = classes - 1;
        }
        swap(rk, nrk);
    }

    // Kasai LCP: lcp[i] = LCP(sa[i], sa[i+1]) for 0 <= i < n-1
    vector<int> lcp(max(0, n - 1), 0);
    for (int i = 0, k = 0; i < n; i++) {
        if (rk[i] == n - 1) { k = 0; continue; }
        int j = sa[rk[i] + 1];
        while (i + k < n && j + k < n && s[i + k] == s[j + k]) k++;
        lcp[rk[i]] = k;
        if (k > 0) k--;
    }

    vector<long long> by_freq(n + 2, 0);
    // Frequency = 1
    for (int i = 0; i < n; i++) {
        int adj_mx = 0;
        if (i > 0) adj_mx = max(adj_mx, lcp[i - 1]);
        if (i + 1 < n) adj_mx = max(adj_mx, lcp[i]);
        by_freq[1] += (n - sa[i]) - adj_mx;
    }

    // Frequency >= 2 via Cartesian intervals (< on left, <= on right)
    int m = n - 1;
    if (m > 0) {
        vector<int> L(m, -1), R(m, m), st;
        st.reserve(m);
        for (int i = 0; i < m; i++) {
            while (!st.empty() && lcp[st.back()] >= lcp[i]) st.pop_back();
            L[i] = st.empty() ? -1 : st.back();
            st.push_back(i);
        }
        st.clear();
        for (int i = m - 1; i >= 0; i--) {
            while (!st.empty() && lcp[st.back()] > lcp[i]) st.pop_back();
            R[i] = st.empty() ? m : st.back();
            st.push_back(i);
        }
        for (int i = 0; i < m; i++) {
            int lo = 0;
            if (L[i] != -1) lo = max(lo, lcp[L[i]]);
            if (R[i] != m) lo = max(lo, lcp[R[i]]);
            if (lcp[i] > lo) {
                int freq = R[i] - L[i];
                by_freq[freq] += (lcp[i] - lo);
            }
        }
    }

    vector<long long> pref(n + 1, 0);
    for (int f = 1; f <= n; f++) pref[f] = pref[f - 1] + by_freq[f];

    while (q--) {
        int a, b;
        cin >> a >> b;
        cout << (pref[b] - pref[a - 1]) << '\n';
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;

    // Double rolling hash, O(N) memory and O(N^2 log N) time by scanning one length at a time
    static constexpr uint64_t M1 = 1'000'000'007ULL, M2 = 1'000'000'009ULL;
    static constexpr uint64_t B1 = 911382323ULL, B2 = 972663749ULL;
    vector<uint64_t> p1(n + 1, 1), p2(n + 1, 1), h1(n + 1, 0), h2(n + 1, 0);
    for (int i = 0; i < n; i++) {
        p1[i + 1] = p1[i] * B1 % M1;
        p2[i + 1] = p2[i] * B2 % M2;
        h1[i + 1] = (h1[i] * B1 + (unsigned char)s[i]) % M1;
        h2[i + 1] = (h2[i] * B2 + (unsigned char)s[i]) % M2;
    }

    vector<long long> by_freq(n + 1, 0);
    vector<uint64_t> cur;
    cur.reserve(n);
    for (int len = 1; len <= n; len++) {
        cur.clear();
        for (int i = 0; i + len <= n; i++) {
            uint64_t x1 = (h1[i + len] + M1 - h1[i] * p1[len] % M1) % M1;
            uint64_t x2 = (h2[i + len] + M2 - h2[i] * p2[len] % M2) % M2;
            cur.push_back((x1 << 32) | x2);
        }
        sort(cur.begin(), cur.end());
        for (size_t i = 0; i < cur.size(); ) {
            size_t j = i + 1;
            while (j < cur.size() && cur[j] == cur[i]) j++;
            by_freq[j - i]++;
            i = j;
        }
    }

    vector<long long> pref(n + 1, 0);
    for (int f = 1; f <= n; f++) pref[f] = pref[f - 1] + by_freq[f];
    while (q--) {
        int a, b;
        cin >> a >> b;
        cout << (pref[b] - pref[a - 1]) << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_65: CENSORED CHRONICLE (2000) - Multi-Pattern Aho-Corasick Stack
    # ------------------------------------------------------------------
    def gen_65(kind: str, seed: int) -> str:
        r = random.Random(seed)
        if kind == "many_patterns_large":
            # 45,000 distinct patterns of length 4 (sum |P_i| = 180,000 <= 200,000) + 200,000 text -> brute K*N = 9e9 checks
            import itertools
            all4 = ["".join(t) for t in itertools.product("abcdefghijklmnop", repeat=4)]
            pats_l = sorted(r.sample(all4, 45000))
            # Construct nested text + random characters
            chunks = []
            tot = 0
            while tot < 195000:
                if r.random() < 0.4:
                    p = r.choice(pats_l)
                    q = r.choice(pats_l)
                    s = p[:2] + q + p[2:]
                else:
                    s = "".join(r.choice("abcdefghij") for _ in range(5))
                chunks.append(s)
                tot += len(s)
            t = "".join(chunks)[:200000]
            return f"{t}\n{len(pats_l)}\n" + "\n".join(pats_l) + "\n"
        elif kind == "many_lengths_large":
            # 580 distinct pattern lengths (sum ~ 168,490) + long periodic text
            pats_l = []
            for ln in range(2, 582):
                pats_l.append("a" * (ln // 2) + "b" + "a" * (ln - 1 - ln // 2) + "c")
            t = ("a" * 300 + "b" + "a" * 290 + "d") * 330
            t = (t + "abac" * 5000)[:200000]
            return f"{t}\n{len(pats_l)}\n" + "\n".join(pats_l) + "\n"
        elif kind == "cascade_large":
            pats_l = ["ab", "cd", "efg", "ababa", "bbaa"]
            while len(pats_l) < 5000:
                pats_l.append("".join(r.choice("abcdef") for _ in range(r.randint(3, 8))))
            pats_l = list(dict.fromkeys(pats_l))
            # Build nested cascades a(ab)b -> ab -> empty
            buf = []
            for _ in range(25000):
                buf.append(r.choice(["aabb", "ccdd", "efg", "abc", "d"]))
            t = "".join(buf)[:200000]
            return f"{t}\n{len(pats_l)}\n" + "\n".join(pats_l) + "\n"
        elif kind == "small_overlap":
            # Includes suffix-overlapping patterns (e.g., "bc" and "abc")
            pats_l = ["bc", "abc", "xy", "axby"]
            t = "aabcxaxbyzabc"
            return f"{t}\n{len(pats_l)}\n" + "\n".join(pats_l) + "\n"
        else:
            pats_l = list({"".join(r.choice("abcd") for _ in range(r.randint(2, 5))) for _ in range(80)})
            t = "".join(r.choice("abcd") for _ in range(900))
            return f"{t}\n{len(pats_l)}\n" + "\n".join(pats_l) + "\n"

    specs.append(ProblemSpec(
        pid="synth_65",
        title="Censored Chronicle",
        rating=2000,
        tags=["strings", "aho-corasick", "automata", "stacks"],
        notes="Build an Aho-Corasick DFA over the K forbidden words propagating max_len along failure links; maintain a stack of (char, state) in O(|T| + 26 * sum|P_i|).",
        story_and_spec="""A royal scribe copies a text T from left to right into an initially empty scroll R, checking a list of K forbidden words P_1, P_2, ..., P_K after each character:
1. Append the next character of T to the end of R.
2. If one or more forbidden words now appear as a suffix of R, immediately erase the LONGEST such forbidden word from the end of R.

After all characters of T have been processed, print the final contents of the scroll R, or print EMPTY if R has length 0.

Input
The first line contains the text T.
The second line contains K.
Each of the next K lines contains a forbidden word P_i.

Output
Print the final string R, or EMPTY if R is empty.

Constraints
1 <= |T| <= 200000
1 <= K <= 100000
1 <= |P_i|, and |P_1| + |P_2| + ... + |P_K| <= 200000
T and all P_i consist of lowercase English letters ('a'-'z')""",
        samples=[
            "abxabyabz\n2\naby\nababz\n",
            "aabb\n1\nab\n",
        ],
        small_inputs=[
            gen_65("small_overlap", 6501),
            gen_65("small_rand", 6502),
            "zzzz\n2\nzz\nz\n",
        ],
        large_inputs=[
            gen_65("many_patterns_large", 6511),
            gen_65("many_lengths_large", 6512),
            gen_65("cascade_large", 6513),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string t;
    int k;
    cin >> t >> k;

    vector<array<int, 26>> tr;
    vector<int> fail, best_len;
    tr.push_back({});
    tr[0].fill(0);
    fail.push_back(0);
    best_len.push_back(0);

    for (int i = 0; i < k; i++) {
        string p; cin >> p;
        int u = 0;
        for (char ch : p) {
            int c = ch - 'a';
            if (!tr[u][c]) {
                tr[u][c] = (int)tr.size();
                tr.push_back({});
                tr.back().fill(0);
                fail.push_back(0);
                best_len.push_back(0);
            }
            u = tr[u][c];
        }
        best_len[u] = max(best_len[u], (int)p.size());
    }

    queue<int> q;
    for (int c = 0; c < 26; c++) {
        if (tr[0][c]) q.push(tr[0][c]);
    }
    while (!q.empty()) {
        int u = q.front(); q.pop();
        best_len[u] = max(best_len[u], best_len[fail[u]]);
        for (int c = 0; c < 26; c++) {
            int v = tr[u][c];
            if (v) {
                fail[v] = tr[fail[u]][c];
                q.push(v);
            } else {
                tr[u][c] = tr[fail[u]][c];
            }
        }
    }

    string res;
    res.reserve(t.size());
    vector<int> st = {0};
    st.reserve(t.size() + 1);

    for (char ch : t) {
        int u = tr[st.back()][ch - 'a'];
        res.push_back(ch);
        st.push_back(u);
        int rem = best_len[u];
        if (rem > 0) {
            res.resize(res.size() - rem);
            st.resize(st.size() - rem);
        }
    }

    if (res.empty()) cout << "EMPTY\n";
    else cout << res << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string t;
    int k;
    cin >> t >> k;
    vector<string> pats(k);
    for (int i = 0; i < k; i++) cin >> pats[i];
    // Sort patterns descending by length so first suffix match is longest
    sort(pats.begin(), pats.end(), [](const string& a, const string& b) {
        return a.size() > b.size();
    });

    string res;
    for (char ch : t) {
        res.push_back(ch);
        int rem = 0;
        for (const string& p : pats) {
            size_t m = p.size();
            if (m <= res.size() && res.compare(res.size() - m, m, p) == 0) {
                rem = (int)m;
                break;
            }
        }
        if (rem > 0) res.resize(res.size() - rem);
    }
    if (res.empty()) cout << "EMPTY\n";
    else cout << res << '\n';
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_13())
