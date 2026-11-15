"""Batch 5 (synth_21 .. synth_25): Strings, Hashing, Persistent Trie, Rightmost XOR Basis, Aho-Corasick, Suffix Array + LCP Stack."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_05() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_21: MIRROR SCROLL (1800) - Dynamic Point-Update Palindrome Queries
    # ------------------------------------------------------------------
    def gen_21(n: int, q: int, seed: int, almost_pal: bool = True) -> str:
        r = random.Random(seed)
        if almost_pal:
            half = [r.choice("ab") for _ in range((n + 1) // 2)]
            s = half + half[: n // 2][::-1]
            for _ in range(max(1, n // 50)):
                s[r.randrange(n)] = r.choice("abc")
        else:
            s = [r.choice("abcd") for _ in range(n)]
        lines = [f"{n} {q}", "".join(s)]
        for i in range(q):
            if i < q - 1 and r.random() < 0.25:
                p = r.randint(1, n)
                c = r.choice("abc")
                lines.append(f"1 {p} {c}")
            else:
                if almost_pal and r.random() < 0.7:
                    d = r.randint(0, n // 3)
                    L, R = 1 + d, n - d
                else:
                    L, R = r.randint(1, n), r.randint(1, n)
                if L > R:
                    L, R = R, L
                lines.append(f"2 {L} {R}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_21",
        title="Mirror Scroll",
        rating=1800,
        tags=["strings", "hashing", "fenwick tree", "data structures"],
        notes="Double-modulus polynomial rolling hash maintained in forward and reverse Fenwick trees.",
        story_and_spec="""An ancient scroll consists of N lowercase English letters indexed 1 to N. Process Q operations of two types:
- 1 P C: replace the character at position P with lowercase letter C.
- 2 L R: check whether the substring from position L to position R (inclusive) reads the same forwards and backwards (is a palindrome).

Input
The first line contains N and Q.
The second line contains the initial string of length N.
Each of the next Q lines describes an operation in one of the two formats above. At least one operation is of type 2.

Output
For each operation of type 2, print YES if the substring is a palindrome, or NO otherwise.

Constraints
1 <= N, Q <= 200000
1 <= P <= N
1 <= L <= R <= N
All characters are lowercase English letters ('a'-'z').""",
        samples=[
            "7 5\nabacaba\n2 1 7\n2 2 5\n1 4 z\n2 1 7\n1 2 c\n",
            "5 3\nabcba\n2 1 5\n1 2 x\n2 1 5\n",
        ],
        small_inputs=[
            "1 3\na\n2 1 1\n1 1 b\n2 1 1\n",
            gen_21(400, 500, 2101, True),
            gen_21(1400, 1500, 2102, True),
            gen_21(1600, 1600, 2103, False),
        ],
        large_inputs=[
            # Keep string all-'a' except center edits so naive two-pointer scans ~100k chars per query!
            (lambda: (
                lambda n, q, r: (
                    f"{n} {q}\n" + ("a" * n) + "\n"
                    + "\n".join(
                        (f"1 {n // 2 + r.randint(-2, 2)} {r.choice('ab')}" if k % 25 == 0
                         else f"2 {r.randint(1, 500)} {n - r.randint(0, 500)}")
                        for k in range(q)
                    ) + "\n"
                )
            )(200000, 200000, random.Random(2111)))(),
            gen_21(190000, 190000, 2112, True),
            gen_21(200000, 200000, 2113, True),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

template<long long MOD, long long BASE>
struct HashBIT {
    int n;
    vector<long long> pw, ipw, bit;

    static long long modpow(long long a, long long e) {
        long long r = 1;
        while (e > 0) {
            if (e & 1) r = r * a % MOD;
            a = a * a % MOD;
            e >>= 1;
        }
        return r;
    }

    HashBIT(int n) : n(n), pw(n + 2, 1), ipw(n + 2, 1), bit(n + 2, 0) {
        long long ib = modpow(BASE, MOD - 2);
        for (int i = 1; i <= n + 1; i++) {
            pw[i] = pw[i - 1] * BASE % MOD;
            ipw[i] = ipw[i - 1] * ib % MOD;
        }
    }
    void add(int idx, long long val) {
        val = (val % MOD + MOD) % MOD;
        for (; idx <= n; idx += idx & -idx) {
            bit[idx] += val;
            if (bit[idx] >= MOD) bit[idx] -= MOD;
        }
    }
    void set_char(int idx, int old_v, int new_v) {
        long long diff = (new_v - old_v) * pw[idx] % MOD;
        add(idx, diff);
    }
    long long sum(int idx) const {
        long long s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s % MOD;
    }
    long long get_hash(int l, int r) const {
        long long raw = (sum(r) - sum(l - 1) + MOD) % MOD;
        return raw * ipw[l] % MOD;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;
    HashBIT<1000000007LL, 137LL> f1(n), r1(n);
    HashBIT<1000000009LL, 139LL> f2(n), r2(n);

    for (int i = 1; i <= n; i++) {
        int v = s[i - 1] - 'a' + 1;
        f1.set_char(i, 0, v);
        r1.set_char(n - i + 1, 0, v);
        f2.set_char(i, 0, v);
        r2.set_char(n - i + 1, 0, v);
    }

    while (q--) {
        int t;
        cin >> t;
        if (t == 1) {
            int p; char c;
            cin >> p >> c;
            int old_v = s[p - 1] - 'a' + 1;
            int new_v = c - 'a' + 1;
            if (old_v != new_v) {
                s[p - 1] = c;
                f1.set_char(p, old_v, new_v);
                r1.set_char(n - p + 1, old_v, new_v);
                f2.set_char(p, old_v, new_v);
                r2.set_char(n - p + 1, old_v, new_v);
            }
        } else {
            int l, r;
            cin >> l >> r;
            bool ok = (f1.get_hash(l, r) == r1.get_hash(n - r + 1, n - l + 1)) &&
                      (f2.get_hash(l, r) == r2.get_hash(n - r + 1, n - l + 1));
            cout << (ok ? "YES\n" : "NO\n");
        }
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
        int t;
        cin >> t;
        if (t == 1) {
            int p; char c;
            cin >> p >> c;
            s[p - 1] = c;
        } else {
            int l, r;
            cin >> l >> r;
            --l; --r;
            bool ok = true;
            while (l < r) {
                if (s[l++] != s[r--]) { ok = false; break; }
            }
            cout << (ok ? "YES\n" : "NO\n");
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_22: SIGNAL INTERFERENCE (1900) - Persistent Binary Trie Max XOR
    # ------------------------------------------------------------------
    def gen_22(n: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        lim = (1 << 30) - 1
        a = [r.randint(0, lim) for _ in range(n)]
        qs = []
        for _ in range(q):
            L, R = r.randint(1, n), r.randint(1, n)
            if L > R:
                L, R = R, L
            if r.random() < 0.6:
                L = r.randint(1, max(1, n // 5))
                R = r.randint(max(L, 4 * n // 5), n)
            qs.append(f"{L} {R} {r.randint(0, lim)}")
        return f"{n} {q}\n" + " ".join(map(str, a)) + "\n" + "\n".join(qs) + "\n"

    specs.append(ProblemSpec(
        pid="synth_22",
        title="Signal Interference",
        rating=1900,
        tags=["data structures", "trie", "persistent", "bitmasks"],
        notes="Persistent 30-bit binary trie storing the maximum array index reaching each node; walk bits 29..0 greedily checking max_idx >= L.",
        story_and_spec="""N radio beacons stand in a line numbered 1 to N. Beacon i broadcasts on frequency code A_i. You are given Q probe queries: each query specifies L, R, and X, and asks for the maximum bitwise XOR value (A_i XOR X) over all beacons i with L <= i <= R.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L, R, and X.

Output
For each query print the maximum value of (A_i XOR X) on its own line.

Constraints
1 <= N, Q <= 180000
0 <= A_i, X < 2^30
1 <= L <= R <= N""",
        samples=[
            "5 3\n3 10 5 25 2\n1 5 5\n1 3 6\n2 4 12\n",
        ],
        small_inputs=[
            "1 2\n7\n1 1 0\n1 1 7\n",
            gen_22(400, 500, 2201),
            gen_22(1500, 1500, 2202),
        ],
        large_inputs=[
            gen_22(170000, 170000, 2211),
            gen_22(180000, 180000, 2212),
            gen_22(180000, 180000, 2213),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int B = 30;
    int max_nodes = (n + 2) * (B + 1);
    vector<array<int, 2>> ch(max_nodes, {0, 0});
    vector<int> mx_pos(max_nodes, 0);
    vector<int> root(n + 1, 0);
    int ptr = 0;

    auto insert = [&](int prev_root, int val, int idx) {
        int nr = ++ptr;
        int u = nr, p = prev_root;
        mx_pos[u] = idx;
        for (int b = B - 1; b >= 0; b--) {
            int bit = (val >> b) & 1;
            ch[u][bit ^ 1] = ch[p][bit ^ 1];
            int nxt = ++ptr;
            ch[u][bit] = nxt;
            u = nxt;
            p = ch[p][bit];
            mx_pos[u] = idx;
        }
        return nr;
    };

    for (int i = 1; i <= n; i++) {
        int x; cin >> x;
        root[i] = insert(root[i - 1], x, i);
    }

    while (q--) {
        int l, r, x;
        cin >> l >> r >> x;
        int u = root[r], ans = 0;
        for (int b = B - 1; b >= 0; b--) {
            int want = ((x >> b) & 1) ^ 1;
            int c = ch[u][want];
            if (c && mx_pos[c] >= l) {
                ans |= (1 << b);
                u = c;
            } else {
                u = ch[u][want ^ 1];
            }
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
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int l, r, x;
        cin >> l >> r >> x;
        int best = 0;
        for (int i = l; i <= r; i++) best = max(best, a[i] ^ x);
        cout << best << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_23: CRYSTAL RESONANCE (2200) - Subarray Max Subset XOR (Rightmost Basis)
    # ------------------------------------------------------------------
    def gen_23(n: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        lim = (1 << 30) - 1
        a = [r.randint(0, lim) for _ in range(n)]
        qs = []
        for _ in range(q):
            if r.random() < 0.7:
                L = r.randint(1, max(1, n // 4))
                R = r.randint(max(L, 3 * n // 4), n)
            else:
                L, R = r.randint(1, n), r.randint(1, n)
                if L > R:
                    L, R = R, L
            qs.append(f"{L} {R}")
        return f"{n} {q}\n" + " ".join(map(str, a)) + "\n" + "\n".join(qs) + "\n"

    specs.append(ProblemSpec(
        pid="synth_23",
        title="Crystal Resonance",
        rating=2200,
        tags=["math", "linear algebra", "bitmasks", "xor basis", "offline"],
        notes="Prefix rightmost XOR linear basis: for each prefix r, store 30 basis vectors maximizing minimum used index; query [L, R] uses vectors with pos >= L in O(30).",
        story_and_spec="""A gallery displays N energy crystals in a row numbered 1 to N, where crystal i has charge A_i. You are asked Q experiments. Each experiment specifies L and R: choose ANY subset of crystals from positions L, L+1, ..., R (possibly the empty subset, whose XOR is 0) to maximize the bitwise XOR of their charges.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L and R.

Output
For each experiment print the maximum achievable subset XOR on its own line.

Constraints
1 <= N, Q <= 200000
0 <= A_i < 2^30
1 <= L <= R <= N""",
        samples=[
            "5 4\n3 5 10 2 8\n1 2\n2 4\n1 5\n4 4\n",
        ],
        small_inputs=[
            "3 2\n0 0 0\n1 3\n2 2\n",
            gen_23(300, 400, 2301),
            gen_23(1000, 1000, 2302),
        ],
        large_inputs=[
            # Small values / linearly dependent prefix so naive basis still has to scan whole [L..R] without early 30-dim exit!
            (lambda: (
                lambda n, q, r: (
                    f"{n} {q}\n"
                    + " ".join(str((r.randint(0, (1 << 15) - 1) << 5)) for _ in range(n)) + "\n"
                    + "\n".join(f"{r.randint(1, n // 6)} {r.randint(5 * n // 6, n)}" for _ in range(q)) + "\n"
                )
            )(200000, 200000, random.Random(2311)))(),
            gen_23(200000, 200000, 2312),
            gen_23(200000, 200000, 2313),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int B = 30;
    vector<array<int, B>> bas(n + 1), pos(n + 1);
    bas[0].fill(0);
    pos[0].fill(0);

    for (int i = 1; i <= n; i++) {
        int x; cin >> x;
        bas[i] = bas[i - 1];
        pos[i] = pos[i - 1];
        int p = i;
        for (int b = B - 1; b >= 0; b--) {
            if ((x >> b) & 1) {
                if (!bas[i][b]) {
                    bas[i][b] = x;
                    pos[i][b] = p;
                    break;
                }
                if (pos[i][b] < p) {
                    swap(bas[i][b], x);
                    swap(pos[i][b], p);
                }
                x ^= bas[i][b];
            }
        }
    }

    while (q--) {
        int l, r;
        cin >> l >> r;
        int ans = 0;
        const auto& bv = bas[r];
        const auto& pv = pos[r];
        for (int b = B - 1; b >= 0; b--) {
            if (pv[b] >= l && (ans ^ bv[b]) > ans) ans ^= bv[b];
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
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int l, r;
        cin >> l >> r;
        int b[30] = {};
        for (int i = l; i <= r; i++) {
            int x = a[i];
            for (int k = 29; k >= 0; k--) if ((x >> k) & 1) {
                if (!b[k]) { b[k] = x; break; }
                x ^= b[k];
            }
        }
        int ans = 0;
        for (int k = 29; k >= 0; k--) if ((ans ^ b[k]) > ans) ans ^= b[k];
        cout << ans << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_24: ARCHIVE PATTERNS (2000) - Aho-Corasick Multi-Pattern Counting
    # ------------------------------------------------------------------
    def gen_24(n: int, q: int, tot_p: int, alpha: str, seed: int) -> str:
        r = random.Random(seed)
        text = "".join(r.choice(alpha) for _ in range(n))
        lines = [text, str(q)]
        rem = tot_p
        for i in range(q):
            left = q - i
            ln = max(1, rem - (left - 1) if i == q - 1 else r.randint(1, max(1, (2 * rem) // left)))
            ln = min(ln, n)
            rem = max(left - 1, rem - ln)
            if r.random() < 0.7:
                st = r.randint(0, n - ln)
                pat = text[st: st + ln]
            else:
                pat = "".join(r.choice(alpha) for _ in range(ln))
            lines.append(pat)
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_24",
        title="Archive Patterns",
        rating=2000,
        tags=["strings", "aho-corasick", "trees", "bfs"],
        notes="Aho-Corasick automaton over all query patterns; run text T once marking visited states, then propagate counts in reverse BFS order along suffix links.",
        story_and_spec="""A royal chronicle is recorded as a string T of lowercase English letters. Scholars submit Q search words P_1, P_2, ..., P_Q (not necessarily distinct). For each word P_i, count how many positions in T match P_i as a contiguous substring (overlapping occurrences are counted separately).

Input
The first line contains the string T.
The second line contains Q.
Each of the next Q lines contains a non-empty string P_i of lowercase English letters.
The sum of lengths of all P_i does not exceed 250000.

Output
Print Q lines: on line i print the number of occurrences of P_i in T.

Constraints
1 <= |T| <= 200000
1 <= Q <= 150000
1 <= |P_i| <= |T|, sum of |P_i| <= 250000""",
        samples=[
            "ababa\n4\na\naba\nab\nbb\n",
        ],
        expected_samples=["3\n2\n2\n0\n"],
        small_inputs=[
            "aaaaa\n3\na\naa\naaaaaa\n"[:0] + "aaaaa\n3\na\naa\naaaaa\n",
            gen_24(600, 400, 1800, "ab", 2401),
            gen_24(1500, 900, 3500, "abc", 2402),
        ],
        large_inputs=[
            # Adversarial 'aaaa...ab' style for naive search + many short/medium patterns
            (lambda: (
                lambda n, q, r: (
                    ("a" * (n - 1) + "b\n") + f"{q}\n"
                    + "\n".join(("a" * r.randint(1, 5) + ("b" if k % 4 == 0 else "a")) for k in range(q)) + "\n"
                )
            )(180000, 80000, random.Random(2411)))(),
            gen_24(180000, 90000, 220000, "ab", 2412),
            gen_24(200000, 100000, 240000, "abcd", 2413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string T;
    int q;
    cin >> T >> q;

    vector<array<int, 26>> nxt(1);
    nxt[0].fill(0);
    vector<int> link(1, 0);
    vector<long long> cnt(1, 0);
    vector<int> end_node(q);

    for (int i = 0; i < q; i++) {
        string p;
        cin >> p;
        int u = 0;
        for (char ch : p) {
            int c = ch - 'a';
            if (!nxt[u][c]) {
                nxt[u][c] = (int)nxt.size();
                nxt.emplace_back();
                nxt.back().fill(0);
                link.push_back(0);
                cnt.push_back(0);
            }
            u = nxt[u][c];
        }
        end_node[i] = u;
    }

    vector<int> bfs;
    bfs.reserve(nxt.size());
    for (int c = 0; c < 26; c++) if (nxt[0][c]) bfs.push_back(nxt[0][c]);
    for (size_t i = 0; i < bfs.size(); i++) {
        int u = bfs[i];
        int f = link[u];
        for (int c = 0; c < 26; c++) {
            int v = nxt[u][c];
            if (v) {
                link[v] = nxt[f][c];
                bfs.push_back(v);
            } else {
                nxt[u][c] = nxt[f][c];
            }
        }
    }

    int u = 0;
    for (char ch : T) {
        u = nxt[u][ch - 'a'];
        cnt[u]++;
    }
    for (int i = (int)bfs.size() - 1; i >= 0; i--) {
        int v = bfs[i];
        cnt[link[v]] += cnt[v];
    }
    for (int i = 0; i < q; i++) cout << cnt[end_node[i]] << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string T;
    int q;
    cin >> T >> q;
    int n = (int)T.size();
    while (q--) {
        string p;
        cin >> p;
        int m = (int)p.size();
        // Naive character scan O(N * M) worst-case, fast on small
        long long ans = 0;
        for (int i = 0; i + m <= n; i++) {
            bool ok = true;
            for (int j = 0; j < m; j++) if (T[i + j] != p[j]) { ok = false; break; }
            if (ok) ans++;
        }
        cout << ans << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_25: TWIN INSCRIPTIONS (2100) - Suffix Array + LCP Stack Common Substrings
    # ------------------------------------------------------------------
    def gen_25(n: int, m: int, alpha: str, seed: int) -> str:
        r = random.Random(seed)
        s = "".join(r.choice(alpha) for _ in range(n))
        t = "".join(r.choice(alpha) for _ in range(m))
        return f"{s}\n{t}\n"

    specs.append(ProblemSpec(
        pid="synth_25",
        title="Twin Inscriptions",
        rating=2100,
        tags=["strings", "suffix array", "lcp", "monotonic stack"],
        notes="Concatenate S + '#' + T, build Suffix Array and Kasai LCP array, then sweep monotonic stacks counting sum of LCPs between suffixes from opposite strings.",
        story_and_spec="""Two stone tablets bear lowercase inscriptions S and T. Count the number of quadruples of 1-based indices (l_1, r_1, l_2, r_2) with 1 <= l_1 <= r_1 <= |S| and 1 <= l_2 <= r_2 <= |T| such that the substring S[l_1 .. r_1] is identical to the substring T[l_2 .. r_2].

Input
The first line contains the non-empty string S.
The second line contains the non-empty string T.

Output
Print a single integer: the total number of matching substring pairs.

Constraints
1 <= |S|, |T| <= 150000
S and T consist of lowercase English letters ('a'-'z').""",
        samples=[
            "aba\nab\n",
        ],
        expected_samples=["4\n"],
        small_inputs=[
            "aaa\naa\n",
            "abc\ndef\n",
            gen_25(400, 450, "ab", 2501),
            gen_25(1200, 1200, "abc", 2502),
        ],
        large_inputs=[
            gen_25(130000, 130000, "ab", 2511),
            gen_25(150000, 150000, "abcd", 2512),
            ("a" * 120000 + "b" * 20000 + "\n" + "a" * 110000 + "b" * 30000 + "\n"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

vector<int> build_sa(const string& s) {
    int n = (int)s.size(), alpha = 256;
    vector<int> p(n), c(n), cnt(max(alpha, n), 0);
    for (int i = 0; i < n; i++) cnt[(unsigned char)s[i]]++;
    for (int i = 1; i < alpha; i++) cnt[i] += cnt[i - 1];
    for (int i = n - 1; i >= 0; i--) p[--cnt[(unsigned char)s[i]]] = i;
    c[p[0]] = 0;
    int classes = 1;
    for (int i = 1; i < n; i++) {
        if (s[p[i]] != s[p[i - 1]]) classes++;
        c[p[i]] = classes - 1;
    }
    vector<int> pn(n), cn(n);
    for (int h = 0; (1 << h) < n; h++) {
        for (int i = 0; i < n; i++) {
            pn[i] = p[i] - (1 << h);
            if (pn[i] < 0) pn[i] += n;
        }
        fill(cnt.begin(), cnt.begin() + classes, 0);
        for (int i = 0; i < n; i++) cnt[c[pn[i]]]++;
        for (int i = 1; i < classes; i++) cnt[i] += cnt[i - 1];
        for (int i = n - 1; i >= 0; i--) p[--cnt[c[pn[i]]]] = pn[i];
        cn[p[0]] = 0;
        classes = 1;
        for (int i = 1; i < n; i++) {
            pair<int, int> cur = {c[p[i]], c[(p[i] + (1 << h)) % n]};
            pair<int, int> prv = {c[p[i - 1]], c[(p[i - 1] + (1 << h)) % n]};
            if (cur != prv) classes++;
            cn[p[i]] = classes - 1;
        }
        c.swap(cn);
        if (classes == n) break;
    }
    return p;
}

vector<int> build_lcp(const string& s, const vector<int>& p) {
    int n = (int)s.size();
    vector<int> rank(n, 0), lcp(n - 1, 0);
    for (int i = 0; i < n; i++) rank[p[i]] = i;
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (rank[i] == n - 1) { k = 0; continue; }
        int j = p[rank[i] + 1];
        while (i + k < n && j + k < n && s[i + k] == s[j + k]) k++;
        lcp[rank[i]] = k;
        if (k) k--;
    }
    return lcp;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string A, B;
    cin >> A >> B;
    int nA = (int)A.size();
    string S = A + "#" + B + "$";
    auto sa = build_sa(S);
    auto lcp = build_lcp(S, sa);
    int m = (int)S.size();

    auto which = [&](int idx) -> int {
        if (idx < nA) return 0;
        if (idx > nA && idx < m - 1) return 1;
        return -1;
    };

    auto count_dir = [&](int src_type, int dst_type) -> long long {
        long long total = 0, cur_sum = 0;
        vector<pair<int, long long>> st; // (lcp_val, count_of_src)
        for (int i = 0; i < m; i++) {
            int tp = which(sa[i]);
            if (tp == dst_type) total += cur_sum;
            if (i + 1 == m) break;
            long long add_cnt = (tp == src_type ? 1 : 0);
            int h = lcp[i];
            while (!st.empty() && st.back().first >= h) {
                cur_sum -= 1LL * st.back().first * st.back().second;
                add_cnt += st.back().second;
                st.pop_back();
            }
            if (add_cnt > 0) {
                st.push_back({h, add_cnt});
                cur_sum += 1LL * h * add_cnt;
            }
        }
        return total;
    };

    cout << (count_dir(0, 1) + count_dir(1, 0)) << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string A, B;
    cin >> A >> B;
    int n = (int)A.size(), m = (int)B.size();
    vector<int> dp(m + 1, 0), ndp(m + 1, 0);
    long long ans = 0;
    for (int i = n - 1; i >= 0; i--) {
        fill(ndp.begin(), ndp.end(), 0);
        for (int j = m - 1; j >= 0; j--) {
            if (A[i] == B[j]) {
                ndp[j] = 1 + dp[j + 1];
                ans += ndp[j];
            }
        }
        dp.swap(ndp);
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_05())
