"""Batch 19 (synth_91 .. synth_95): Grandmaster Wave III (CF 2700-3000) — SAM + Endpos SegTree Merge Substring Prefix Match, Eertree Series-Difference Palindrome Partitioning, CDQ Divide-and-Conquer NTT, Sprague-Grundy + FWHT Nimber Convolution, LGV Lemma + Obstacle Inclusion-Exclusion Disjoint Grid Paths."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_19() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_91: CHRONICLE SUBSTRING SPAN (2900) - SAM + Link-Tree Lifting + SegTree Merge
    # ------------------------------------------------------------------
    def gen_91(n: int, q: int, alpha: int, seed: int) -> str:
        r = random.Random(seed)
        ab = "abcdefghijklmnopqrstuvwxyz"[:alpha]
        # Mix periodic and random blocks so long matches occur frequently
        chars = [r.choice(ab) for _ in range(n)]
        for _ in range(max(3, n // 60)):
            src = r.randrange(max(1, n - 50))
            dst = r.randrange(max(1, n - 50))
            ln = min(60, n - max(src, dst))
            chars[dst:dst + ln] = chars[src:src + ln]
        s = "".join(chars)
        lines = [f"{n} {q}", s]
        for _ in range(q):
            if r.random() < 0.7:
                l = r.randint(1, max(1, n // 4))
                rr = r.randint(max(l, 3 * n // 4), n)
            else:
                l, rr = r.randint(1, n), r.randint(1, n)
                if l > rr:
                    l, rr = rr, l
            pl, pr = r.randint(1, n), r.randint(1, n)
            if pl > pr:
                pl, pr = pr, pl
            lines.append(f"{l} {rr} {pl} {pr}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_91",
        title="Chronicle Substring Span",
        rating=2900,
        tags=["strings", "suffix automaton", "segment tree merge", "binary search", "trees"],
        notes="Build Suffix Automaton of S + Segment Tree Merge of endpos sets over the suffix-link tree + binary lifting; for query (L, R, P_l, P_r), binary search prefix length k checking endpos in [L + k - 1, R] in O(log^2 N).",
        story_and_spec="""You are given a string S of length N (1-indexed) and Q queries. Each query specifies four integers L, R, A, and B (1 <= L <= R <= N, 1 <= A <= B <= N):

Find the maximum integer k (0 <= k <= B - A + 1) such that the length-k prefix of S[A..B] — namely S[A .. A + k - 1] — appears as a contiguous substring completely inside S[L..R] (k = 0 is always valid as the empty string).

Input
The first line contains N and Q.
The second line contains S.
Each of the next Q lines contains four integers L, R, A, and B.

Output
For each query print the maximum valid prefix length k on its own line.

Constraints
1 <= N, Q <= 90000
S consists of lowercase English letters ('a'-'z')
1 <= L <= R <= N
1 <= A <= B <= N""",
        samples=[
            "10 4\nabacabadab\n1 7 4 10\n2 6 1 5\n8 10 1 4\n5 9 8 10\n",
        ],
        small_inputs=[
            "5 2\naaaaa\n1 3 2 5\n4 4 1 5\n",
            gen_91(300, 350, 3, 9101),
            gen_91(1400, 1400, 2, 9102),
        ],
        large_inputs=[
            gen_91(90000, 90000, 2, 9111),
            gen_91(90000, 90000, 4, 9112),
            gen_91(90000, 90000, 8, 9113),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct SAMNode {
    int nxt[26];
    int link = -1, len = 0;
    SAMNode() { memset(nxt, -1, sizeof(nxt)); }
};

struct SegNode {
    int ls = 0, rs = 0;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;

    vector<SAMNode> sam(1);
    sam.reserve(2 * n + 5);
    vector<int> pref_state(n + 1, 0);
    int last = 0;

    for (int i = 0; i < n; i++) {
        int c = s[i] - 'a';
        int cur = (int)sam.size();
        sam.emplace_back();
        sam[cur].len = sam[last].len + 1;
        int p = last;
        while (p != -1 && sam[p].nxt[c] == -1) {
            sam[p].nxt[c] = cur;
            p = sam[p].link;
        }
        if (p == -1) {
            sam[cur].link = 0;
        } else {
            int qn = sam[p].nxt[c];
            if (sam[p].len + 1 == sam[qn].len) {
                sam[cur].link = qn;
            } else {
                int clone = (int)sam.size();
                sam.push_back(sam[qn]);
                sam[clone].len = sam[p].len + 1;
                while (p != -1 && sam[p].nxt[c] == qn) {
                    sam[p].nxt[c] = clone;
                    p = sam[p].link;
                }
                sam[qn].link = sam[cur].link = clone;
            }
        }
        last = cur;
        pref_state[i + 1] = cur;
    }

    int V = (int)sam.size();
    // Counting sort states by len for bottom-up link-tree processing
    vector<int> cnt(n + 1, 0), ord(V);
    for (int i = 0; i < V; i++) cnt[sam[i].len]++;
    for (int i = 1; i <= n; i++) cnt[i] += cnt[i - 1];
    for (int i = V - 1; i >= 0; i--) ord[--cnt[sam[i].len]] = i;

    int LOG = __lg(V) + 2;
    vector<vector<int>> up(LOG, vector<int>(V, 0));
    for (int idx = 1; idx < V; idx++) {
        int u = ord[idx];
        up[0][u] = max(0, sam[u].link);
        for (int b = 1; b < LOG; b++) up[b][u] = up[b - 1][up[b - 1][u]];
    }

    // Persistent (non-destructive!) Segment Tree Merge so every state u retains its own endpos tree
    vector<SegNode> tr(1);
    tr.reserve(n * 40 + 5);
    auto new_seg = [&]() -> int {
        tr.push_back({});
        return (int)tr.size() - 1;
    };

    function<int(int, int, int)> make_pt = [&](int l, int r, int pos) -> int {
        int u = new_seg();
        if (l == r) return u;
        int mid = (l + r) >> 1;
        if (pos <= mid) tr[u].ls = make_pt(l, mid, pos);
        else tr[u].rs = make_pt(mid + 1, r, pos);
        return u;
    };
    function<int(int, int)> merge_seg = [&](int a, int b) -> int {
        if (!a || !b) return a ? a : b;
        int u = new_seg();
        tr[u].ls = merge_seg(tr[a].ls, tr[b].ls);
        tr[u].rs = merge_seg(tr[a].rs, tr[b].rs);
        return u;
    };
    function<bool(int, int, int, int, int)> has_in = [&](int u, int l, int r, int ql, int qr) -> bool {
        if (!u || ql > qr) return false;
        if (ql <= l && r <= qr) return true;
        int mid = (l + r) >> 1;
        if (ql <= mid && has_in(tr[u].ls, l, mid, ql, qr)) return true;
        if (qr > mid && has_in(tr[u].rs, mid + 1, r, ql, qr)) return true;
        return false;
    };

    vector<int> root(V, 0);
    for (int i = 1; i <= n; i++) {
        root[pref_state[i]] = merge_seg(root[pref_state[i]], make_pt(1, n, i));
    }
    for (int idx = V - 1; idx >= 1; idx--) {
        int u = ord[idx];
        int p = sam[u].link;
        if (p >= 0) root[p] = merge_seg(root[p], root[u]);
    }

    auto state_for_sub = [&](int a, int k) -> int {
        int u = pref_state[a + k - 1];
        for (int b = LOG - 1; b >= 0; b--) {
            int p = up[b][u];
            if (sam[p].len >= k) u = p;
        }
        return u;
    };

    while (q--) {
        int l, r, a, b;
        cin >> l >> r >> a >> b;
        int lo = 0, hi = min(r - l + 1, b - a + 1);
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            int st_u = state_for_sub(a, mid);
            if (has_in(root[st_u], 1, n, l + mid - 1, r)) lo = mid;
            else hi = mid - 1;
        }
        cout << lo << '\n';
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

// 40-pt subtask solver: per-query Z-algorithm on S[A..B] + '#' + S[L..R] in O(Q * N)
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;
    string t;
    vector<int> z;
    while (q--) {
        int l, r, a, b;
        cin >> l >> r >> a >> b;
        int plen = b - a + 1, tlen = r - l + 1;
        t.clear();
        t.append(s, a - 1, plen);
        t.push_back('#');
        t.append(s, l - 1, tlen);
        int m = (int)t.size();
        z.assign(m, 0);
        int best = 0;
        for (int i = 1, zl = 0, zr = 0; i < m; i++) {
            if (i <= zr) z[i] = min(zr - i + 1, z[i - zl]);
            while (i + z[i] < m && t[z[i]] == t[i + z[i]]) z[i]++;
            if (i + z[i] - 1 > zr) { zl = i; zr = i + z[i] - 1; }
            if (i > plen) best = max(best, z[i]);
        }
        cout << best << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_92: EVEN PALINDROME SHREDDING (2900) - Eertree Series-Difference DP
    # ------------------------------------------------------------------
    def gen_92(n: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        if kind == "all_a":
            chars = ["a"] * n
            for _ in range(max(2, n // 4000)):
                pos = r.randint(1, max(1, n - 3))
                chars[pos] = "b"
                chars[min(n - 1, pos + 1)] = "b"
            return "".join(chars) + "\n"
        elif kind == "pal_blocks":
            buf = []
            while len(buf) < n:
                half = [r.choice("ab") for _ in range(r.randint(1, 18))]
                blk = half + ( [r.choice("ab")] if r.random() < 0.4 else [] ) + half[::-1]
                buf.extend(blk)
            return "".join(buf[:n]) + "\n"
        else:
            return "".join(r.choice("abc") for _ in range(n)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_92",
        title="Palindrome Shredding",
        rating=2900,
        tags=["strings", "palindromic tree", "eertree", "dp"],
        notes="Eertree (Palindromic Automaton) with series-link (slink, diff) partitioning DP in O(N log N): sum dp over all palindromic suffixes via <= log2(N) arithmetic progressions, minus dp[i-1] to exclude length-1 pieces.",
        story_and_spec="""Given a lowercase string S of length N, count the number of ways to partition S into one or more contiguous substrings such that EVERY piece in the partition:
1. is a palindrome, AND
2. has length AT LEAST 2.

Print the number of valid partitions modulo 998244353.

Input
The single line of input contains the string S (1 <= |S| <= 200000).

Output
Print a single integer: the number of valid palindromic partitions of S with all piece lengths >= 2, modulo 998244353.

Constraints
1 <= |S| <= 200000
S consists of lowercase English letters ('a'-'z')""",
        samples=[
            "ababa\n",
            "aaaa\n",
            "abcba\n",
        ],
        small_inputs=[
            "a\n",
            "aa\n",
            gen_92(400, 9201, "pal_blocks"),
            # N=3500: naive O(N^3) or 2^N gets 0 pts; Manacher + O(N^2) DP gets 40 pts!
            gen_92(3500, 9202, "all_a"),
        ],
        large_inputs=[
            gen_92(200000, 9211, "all_a"),
            gen_92(200000, 9212, "pal_blocks"),
            gen_92(200000, 9213, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 998244353;

struct Node {
    int nxt[26];
    int len = 0, fail = 0, diff = 0, slink = 0;
    Node() { memset(nxt, 0, sizeof(nxt)); }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string s;
    cin >> s;
    int n = (int)s.size();

    // Node 0: len = -1 (odd root), Node 1: len = 0 (even root)
    vector<Node> tr(2);
    tr.reserve(n + 5);
    tr[0].len = -1; tr[0].fail = 0;
    tr[1].len = 0;  tr[1].fail = 0;
    int last = 1;

    string pad = "#" + s;
    auto get_fail = [&](int u, int pos) {
        while (pad[pos - tr[u].len - 1] != pad[pos]) u = tr[u].fail;
        return u;
    };

    vector<int> dp(n + 1, 0), g(n + 5, 0);
    dp[0] = 1;

    for (int i = 1; i <= n; i++) {
        int c = pad[i] - 'a';
        int cur = get_fail(last, i);
        if (!tr[cur].nxt[c]) {
            int nw = (int)tr.size();
            tr.emplace_back();
            tr[nw].len = tr[cur].len + 2;
            int f = tr[get_fail(tr[cur].fail, i)].nxt[c];
            tr[nw].fail = (tr[nw].len == 1 ? 1 : f);
            tr[nw].diff = tr[nw].len - tr[tr[nw].fail].len;
            tr[nw].slink = (tr[nw].diff == tr[tr[nw].fail].diff)
                           ? tr[tr[nw].fail].slink
                           : tr[nw].fail;
            tr[cur].nxt[c] = nw;
        }
        last = tr[cur].nxt[c];

        long long ways_all = 0;
        for (int v = last; tr[v].len > 0; v = tr[v].slink) {
            g[v] = dp[i - tr[tr[v].slink].len - tr[v].diff];
            if (tr[v].diff == tr[tr[v].fail].diff) {
                g[v] += g[tr[v].fail];
                if (g[v] >= MOD) g[v] -= MOD;
            }
            ways_all += g[v];
        }
        // Subtract the unique length-1 palindromic suffix pad[i..i]
        long long ways_ge2 = (ways_all - dp[i - 1]) % MOD;
        if (ways_ge2 < 0) ways_ge2 += MOD;
        dp[i] = (int)ways_ge2;
    }

    cout << dp[n] << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 998244353;

// 40-pt subtask solver: Manacher O(N) + O(N^2) DP
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string s;
    cin >> s;
    int n = (int)s.size();

    vector<int> d1(n), d2(n);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
        while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) k++;
        d1[i] = k--;
        if (i + k > r) { l = i - k; r = i + k; }
    }
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 0 : min(d2[l + r - i + 1], r - i + 1);
        while (0 <= i - k - 1 && i + k < n && s[i - k - 1] == s[i + k]) k++;
        d2[i] = k--;
        if (i + k > r) { l = i - k - 1; r = i + k; }
    }
    auto is_pal = [&](int l, int r) -> bool {
        int len = r - l + 1;
        int mid = (l + r) >> 1;
        if (len & 1) return d1[mid] >= (len + 1) / 2;
        return d2[mid + 1] >= len / 2;
    };

    vector<int> dp(n + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= n; i++) {
        long long tot = 0;
        for (int j = 0; j <= i - 2; j++) {
            if (dp[j] && is_pal(j, i - 1)) tot += dp[j];
        }
        dp[i] = (int)(tot % MOD);
    }
    cout << dp[n] << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_93: CATALAN SIGNAL ECHOES (2700) - CDQ Divide-and-Conquer NTT
    # ------------------------------------------------------------------
    def gen_93(n: int, seed: int) -> str:
        r = random.Random(seed)
        MOD = 998244353
        a = [r.randint(1, MOD - 1) for _ in range(n)]
        w = [r.randint(0, MOD - 1) for _ in range(n)]
        return f"{n}\n" + " ".join(map(str, a)) + "\n" + " ".join(map(str, w)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_93",
        title="Amplified Signal Echoes",
        rating=2700,
        tags=["fft", "ntt", "divide and conquer", "math", "dp"],
        notes="Semi-online non-autonomous convolution F_i = A_i * sum_{j=0}^{i-1} F_j * W_{i-j} mod 998244353: site-varying A_i defeats polynomial inverse 1/(1 - W), requiring CDQ Divide-and-Conquer + NTT in O(N log^2 N).",
        story_and_spec="""A deep-space relay line has stations 0, 1, ..., N. Station 0 emits a unit pulse F_0 = 1.

Each subsequent station i (1 <= i <= N) has gain factor A_i, and an echo jumping k steps forward (k >= 1) carries kernel weight W_k. Station i outputs pulse amplitude:
F_i = ( A_i * sum_{j = 0}^{i - 1} F_j * W_{i - j} ) mod 998244353.

Given A_1 ... A_N and W_1 ... W_N, compute F_1, F_2, ..., F_N modulo 998244353.

Input
The first line contains N.
The second line contains A_1 ... A_N.
The third line contains W_1 ... W_N.

Output
Print N space-separated integers: F_1, F_2, ..., F_N modulo 998244353.

Constraints
1 <= N <= 120000
0 <= A_i, W_i < 998244353""",
        samples=[
            "4\n2 3 1 5\n1 4 2 3\n",
        ],
        small_inputs=[
            "1\n7\n9\n",
            gen_93(300, 9301),
            gen_93(2200, 9302),
        ],
        large_inputs=[
            gen_93(120000, 9311),
            gen_93(120000, 9312),
            gen_93(100000, 9313),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 998244353;
static constexpr int G = 3;

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

void ntt(vector<int>& a, bool invert) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        long long wlen = modpow(G, (MOD - 1) / len);
        if (invert) wlen = modpow(wlen, MOD - 2);
        for (int i = 0; i < n; i += len) {
            long long w = 1;
            int half = len >> 1;
            for (int j = 0; j < half; j++) {
                int u = a[i + j];
                int v = (int)(a[i + j + half] * w % MOD);
                a[i + j] = u + v < MOD ? u + v : u + v - MOD;
                a[i + j + half] = u - v >= 0 ? u - v : u - v + MOD;
                w = w * wlen % MOD;
            }
        }
    }
    if (invert) {
        long long inv_n = modpow(n, MOD - 2);
        for (int& x : a) x = (int)(x * inv_n % MOD);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> A(n + 1, 0), W(n + 1, 0), sum_conv(n + 1, 0), F(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> A[i];
    for (int i = 1; i <= n; i++) cin >> W[i];
    F[0] = 1;

    vector<int> fa, fb;
    function<void(int, int)> cdq = [&](int l, int r) {
        if (l == r) {
            if (l > 0) F[l] = (sum_conv[l] % MOD) * A[l] % MOD;
            return;
        }
        int mid = (l + r) >> 1;
        cdq(l, mid);

        int lenL = mid - l + 1;
        int lenW = r - l;
        int sz = 1;
        while (sz < lenL + lenW) sz <<= 1;
        fa.assign(sz, 0);
        fb.assign(sz, 0);
        for (int i = 0; i < lenL; i++) fa[i] = (int)F[l + i];
        for (int k = 1; k <= lenW; k++) fb[k - 1] = (int)W[k];

        ntt(fa, false);
        ntt(fb, false);
        for (int i = 0; i < sz; i++) fa[i] = (int)(1LL * fa[i] * fb[i] % MOD);
        ntt(fa, true);

        // fa[idx] represents sum_{i + (k - 1) = idx} F[l + i] * W[k] -> target pos = l + idx + 1
        for (int pos = mid + 1; pos <= r; pos++) {
            int idx = pos - l - 1;
            sum_conv[pos] += fa[idx];
            if (sum_conv[pos] >= MOD) sum_conv[pos] -= MOD;
        }

        cdq(mid + 1, r);
    };

    cdq(0, n);
    for (int i = 1; i <= n; i++) {
        cout << F[i] << (i == n ? '\n' : ' ');
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> A(n + 1), W(n + 1), F(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> A[i];
    for (int i = 1; i <= n; i++) cin >> W[i];
    F[0] = 1;
    for (int i = 1; i <= n; i++) {
        unsigned __int128 acc = 0;
        for (int j = 0; j < i; j++) acc += (unsigned __int128)F[j] * W[i - j];
        F[i] = (long long)((acc % MOD) * A[i] % MOD);
    }
    for (int i = 1; i <= n; i++) cout << F[i] << (i == n ? '\n' : ' ');
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_94: NIMBLE CRYSTAL HEAPS (2700) - Sprague-Grundy + XOR FWHT Exponentiation
    # ------------------------------------------------------------------
    def gen_94(v: int, e: int, k: int, bbits: int, seed: int) -> str:
        r = random.Random(seed)
        b = [r.randint(0, (1 << bbits) - 1) for _ in range(v)]
        edges = set()
        tries = 0
        while len(edges) < e and tries < e * 5:
            tries += 1
            x, y = r.randint(1, v), r.randint(1, v)
            if x != y:
                if x > y:
                    x, y = y, x
                edges.add((x, y))
        lines = [f"{v} {len(edges)} {k}", " ".join(map(str, b))]
        for x, y in edges:
            lines.append(f"{x} {y}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_94",
        title="Nimble Crystal Heaps",
        rating=2700,
        tags=["games", "sprague-grundy", "fwht", "fft", "bitmasks"],
        notes="Sprague-Grundy theorem gives per-station nimber n_u = mex_DAG(u) XOR B_u (< 2^18); count K-tuples with non-zero XOR via Fast Walsh-Hadamard Transform + modular exponentiation K <= 10^18.",
        story_and_spec="""A directed acyclic cavern network has V chambers numbered 1 to V and E one-way tunnels (every tunnel goes from a smaller-numbered chamber to a larger-numbered chamber). Each chamber u also has a crystal rating B_u.

An expedition spawns K independent units by choosing an ordered K-tuple of chambers (u_1, u_2, ..., u_K) from {1, ..., V}^K (chambers may repeat). Spawning unit j at chamber u_j places BOTH:
- one explorer token at chamber u_j, AND
- one pile containing B_{u_j} crystals.

Alice and Bob then play an impartial game, with Alice moving first. On a turn, the active player chooses EITHER:
- one explorer token that has at least one outgoing tunnel from its current chamber and moves it along one outgoing tunnel, OR
- one non-empty crystal pile and removes any positive number of crystals (from 1 up to the entire pile).

The player who has no legal move loses. Assuming optimal play, how many of the V^K possible initial K-tuples (u_1, ..., u_K) are winning for Alice, modulo 998244353?

Input
The first line contains V, E, and K.
The second line contains B_1 ... B_V.
Each of the next E lines contains two integers X_i and Y_i (1 <= X_i < Y_i <= V), representing a directed tunnel X_i -> Y_i.

Output
Print the number of Alice-winning K-tuples modulo 998244353.

Constraints
1 <= V <= 100000
0 <= E <= 200000
1 <= K <= 10^18
0 <= B_u < 2^17""",
        samples=[
            "3 2 2\n0 1 2\n1 2\n2 3\n",
            "4 3 3\n3 1 0 2\n1 2\n1 3\n2 3\n",
        ],
        small_inputs=[
            "2 0 1\n0 5\n",
            gen_94(120, 300, 15, 8, 9401),
            gen_94(250, 700, 25, 9, 9402),
        ],
        large_inputs=[
            # large[0] keeps K=400, B<2^11 so O(K * M^2) XOR-DP cleanly TLEs (~1.6e9 ops) without OOM!
            gen_94(100000, 200000, 400, 11, 9411),
            gen_94(100000, 200000, 10**18, 17, 9412),
            gen_94(100000, 200000, 987654321012345678, 17, 9413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

long long modpow(long long a, long long e) {
    a %= MOD;
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int v, e;
    long long K;
    cin >> v >> e >> K;
    vector<int> B(v + 1);
    for (int i = 1; i <= v; i++) cin >> B[i];
    vector<vector<int>> g(v + 1);
    for (int i = 0; i < e; i++) {
        int x, y; cin >> x >> y;
        g[x].push_back(y);
    }

    vector<int> gr(v + 1, 0), seen(v + 5, 0);
    int stamp = 0;
    for (int u = v; u >= 1; u--) {
        ++stamp;
        for (int nb : g[u]) if (gr[nb] < (int)seen.size()) seen[gr[nb]] = stamp;
        int mex = 0;
        while (seen[mex] == stamp) mex++;
        gr[u] = mex;
    }

    int max_val = 0;
    for (int u = 1; u <= v; u++) max_val = max(max_val, gr[u] ^ B[u]);
    int SZ = 1;
    while (SZ <= max_val) SZ <<= 1;

    vector<long long> poly(SZ, 0);
    for (int u = 1; u <= v; u++) poly[gr[u] ^ B[u]]++;

    // XOR Fast Walsh-Hadamard Transform
    for (int len = 1; len < SZ; len <<= 1) {
        for (int i = 0; i < SZ; i += (len << 1)) {
            for (int j = 0; j < len; j++) {
                long long x = poly[i + j], y = poly[i + j + len];
                poly[i + j] = (x + y) % MOD;
                poly[i + j + len] = (x - y + MOD) % MOD;
            }
        }
    }

    for (int i = 0; i < SZ; i++) poly[i] = modpow(poly[i], K);

    // Inverse XOR FWHT: only need poly[0] = (1 / SZ) * sum_i poly[i]!
    long long sum_hat = 0;
    for (int i = 0; i < SZ; i++) sum_hat = (sum_hat + poly[i]) % MOD;
    long long zero_tuples = sum_hat * modpow(SZ, MOD - 2) % MOD;
    long long total_tuples = modpow(v, K);
    cout << (total_tuples - zero_tuples + MOD) % MOD << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

// 40-pt subtask solver: DAG Grundy + O(K * SZ^2) direct XOR convolution DP
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int v, e;
    long long K;
    cin >> v >> e >> K;
    vector<int> B(v + 1);
    for (int i = 1; i <= v; i++) cin >> B[i];
    vector<vector<int>> g(v + 1);
    for (int i = 0; i < e; i++) {
        int x, y; cin >> x >> y;
        g[x].push_back(y);
    }

    vector<int> gr(v + 1, 0), seen(v + 5, 0);
    int stamp = 0;
    for (int u = v; u >= 1; u--) {
        ++stamp;
        for (int nb : g[u]) seen[gr[nb]] = stamp;
        int mex = 0;
        while (seen[mex] == stamp) mex++;
        gr[u] = mex;
    }

    int SZ = 1;
    for (int u = 1; u <= v; u++) while (SZ <= (gr[u] ^ B[u])) SZ <<= 1;
    vector<long long> cnt(SZ, 0), dp(SZ, 0), ndp(SZ, 0);
    for (int u = 1; u <= v; u++) cnt[gr[u] ^ B[u]]++;
    dp[0] = 1;
    for (long long step = 0; step < K; step++) {
        fill(ndp.begin(), ndp.end(), 0);
        for (int a = 0; a < SZ; a++) if (dp[a]) {
            for (int b = 0; b < SZ; b++) if (cnt[b]) {
                ndp[a ^ b] = (ndp[a ^ b] + dp[a] * cnt[b]) % MOD;
            }
        }
        dp.swap(ndp);
    }
    long long tot = 0;
    for (int x = 1; x < SZ; x++) tot = (tot + dp[x]) % MOD;
    cout << tot << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_95: DISJOINT CARAVAN ROUTES (3000) - LGV Lemma + Obstacle Inclusion-Exclusion
    # ------------------------------------------------------------------
    def gen_95(h: int, w: int, k: int, n_obs: int, seed: int) -> str:
        r = random.Random(seed)
        starts = sorted(r.sample(range(1, w + 1), k))
        ends = sorted(r.sample(range(1, w + 1), k))
        # Ensure for most i, starts[i] <= ends[i] so non-zero tuples exist
        for i in range(k):
            if starts[i] > ends[i]:
                starts[i], ends[i] = ends[i], starts[i]
        starts = sorted(set(starts))
        ends = sorted(set(ends))
        k = min(len(starts), len(ends))
        # Re-pick clean strictly increasing starts <= ends
        pts_s = sorted(r.sample(range(1, w - k + 1), k))
        pts_t = sorted(r.sample(range(1, w + 1), k))
        for i in range(k):
            if pts_s[i] > pts_t[i]:
                pts_s[i], pts_t[i] = pts_t[i], pts_s[i]
        # Deduplicate if swaps created ties
        if len(set(pts_s)) < k or len(set(pts_t)) < k:
            pts_s = sorted(r.sample(range(1, max(k + 1, w // 2 + 1)), k))
            pts_t = sorted(r.sample(range(max(1, w - max(k + 1, w // 2) + 1), w + 1), k))

        forb = set()
        for c in pts_s:
            forb.add((1, c))
        for c in pts_t:
            forb.add((h, c))
        obs = []
        tries = 0
        while len(obs) < n_obs and tries < n_obs * 10:
            tries += 1
            rr, cc = r.randint(1, h), r.randint(1, w)
            if (rr, cc) not in forb:
                forb.add((rr, cc))
                obs.append((rr, cc))
        lines = [
            f"{h} {w} {k} {len(obs)}",
            " ".join(map(str, pts_s)),
            " ".join(map(str, pts_t)),
        ] + [f"{rr} {cc}" for rr, cc in obs]
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_95",
        title="Disjoint Caravan Routes",
        rating=3000,
        tags=["combinatorics", "matrices", "lgv lemma", "dp", "inclusion-exclusion"],
        notes="Two-layer Grandmaster synthesis: (1) Lindstrom-Gessel-Viennot Lemma reduces K vertex-disjoint Right/Down grid paths (1, S_i) -> (H, T_i) to det(M), and (2) M_{i,j} on a 100,000 x 100,000 grid with B <= 180 blocked cells is computed via O(B^2 + K*B) obstacle Inclusion-Exclusion DP!",
        story_and_spec="""A desert forms an H x W grid of cells (rows 1..H from north to south, columns 1..W from west to east). Exactly B cells contain impassable sinkholes.

There are K royal caravans: caravan i starts at cell (1, S_i) on the northern border (with 1 <= S_1 < S_2 < ... < S_K <= W) and must end at cell (H, T_i) on the southern border (with 1 <= T_1 < T_2 < ... < T_K <= W). None of the start or destination cells contain sinkholes.

Each caravan moves only SOUTH ((r, c) -> (r + 1, c)) or EAST ((r, c) -> (r, c + 1)), never entering a sinkhole. Count the number of K-tuples of routes (P_1, P_2, ..., P_K) such that NO TWO caravans ever visit the same grid cell (not even at different times), modulo 998244353.

Input
The first line contains H, W, K, and B.
The second line contains S_1 ... S_K.
The third line contains T_1 ... T_K.
Each of the next B lines contains two integers r_m and c_m, the coordinates of a sinkhole.

Output
Print the number of pairwise vertex-disjoint route K-tuples modulo 998244353.

Constraints
2 <= H, W <= 100000
1 <= K <= 80, K <= W
0 <= B <= 180
1 <= S_1 < S_2 < ... < S_K <= W
1 <= T_1 < T_2 < ... < T_K <= W""",
        samples=[
            "3 3 2 0\n1 2\n2 3\n",
            "4 4 2 1\n1 2\n3 4\n2 3\n",
        ],
        small_inputs=[
            "2 3 2 1\n1 2\n2 3\n1 3\n",
            # Even 40-pt subtask has K=30 on 250x250 so non-LGV exponential search gets 0 pts!
            gen_95(200, 220, 25, 40, 9501),
            gen_95(320, 320, 35, 60, 9502),
        ],
        large_inputs=[
            # large[0]: H=W=60000, K=60 -> brute O(K * H * W) = 2.16e11 ops (1D row DP uses <1 MB RAM, cleanly TLEs!)
            gen_95(60000, 60000, 60, 150, 9511),
            gen_95(100000, 100000, 80, 180, 9512),
            gen_95(100000, 100000, 70, 160, 9513),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int H, W, K, B;
    cin >> H >> W >> K >> B;
    vector<int> S(K), T(K);
    for (int i = 0; i < K; i++) cin >> S[i];
    for (int i = 0; i < K; i++) cin >> T[i];

    vector<pair<int, int>> obs(B);
    for (int i = 0; i < B; i++) cin >> obs[i].first >> obs[i].second;
    sort(obs.begin(), obs.end());

    int MAXF = H + W + 5;
    vector<long long> fac(MAXF, 1), ifac(MAXF, 1);
    for (int i = 1; i < MAXF; i++) fac[i] = fac[i - 1] * i % MOD;
    ifac[MAXF - 1] = modpow(fac[MAXF - 1], MOD - 2);
    for (int i = MAXF - 2; i >= 0; i--) ifac[i] = ifac[i + 1] * (i + 1) % MOD;

    auto ways = [&](int r1, int c1, int r2, int c2) -> long long {
        int dr = r2 - r1, dc = c2 - c1;
        if (dr < 0 || dc < 0) return 0;
        return fac[dr + dc] * ifac[dr] % MOD * ifac[dc] % MOD;
    };

    // Precompute obstacle-to-obstacle free path counts
    vector<vector<long long>> oo(B, vector<long long>(B, 0));
    for (int a = 0; a < B; a++) {
        for (int b = a + 1; b < B; b++) {
            oo[a][b] = ways(obs[a].first, obs[a].second, obs[b].first, obs[b].second);
        }
    }

    // Build K x K LGV path matrix M
    vector<vector<long long>> M(K, vector<long long>(K, 0));
    vector<long long> dp_obs(B);

    for (int i = 0; i < K; i++) {
        // dp_obs[a] = paths from (1, S[i]) to obs[a] avoiding all earlier obstacles
        for (int a = 0; a < B; a++) {
            long long val = ways(1, S[i], obs[a].first, obs[a].second);
            for (int p = 0; p < a; p++) if (dp_obs[p] && oo[p][a]) {
                val = (val - dp_obs[p] * oo[p][a]) % MOD;
            }
            if (val < 0) val += MOD;
            dp_obs[a] = val;
        }
        for (int j = 0; j < K; j++) {
            long long val = ways(1, S[i], H, T[j]);
            for (int a = 0; a < B; a++) if (dp_obs[a]) {
                val = (val - dp_obs[a] * ways(obs[a].first, obs[a].second, H, T[j])) % MOD;
            }
            if (val < 0) val += MOD;
            M[i][j] = val;
        }
    }

    // Determinant of M mod 998244353
    long long det = 1;
    for (int col = 0; col < K; col++) {
        int piv = -1;
        for (int row = col; row < K; row++) if (M[row][col] != 0) { piv = row; break; }
        if (piv == -1) { det = 0; break; }
        if (piv != col) {
            swap(M[piv], M[col]);
            det = (MOD - det) % MOD;
        }
        det = det * M[col][col] % MOD;
        long long inv_p = modpow(M[col][col], MOD - 2);
        for (int row = col + 1; row < K; row++) if (M[row][col] != 0) {
            long long factor = M[row][col] * inv_p % MOD;
            for (int c = col; c < K; c++) {
                M[row][c] = (M[row][c] - factor * M[col][c]) % MOD;
                if (M[row][c] < 0) M[row][c] += MOD;
            }
        }
    }

    cout << det << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

// 40-pt subtask solver: LGV Lemma + K rolling-row O(H * W) grid DPs (O(W) RAM, cleanly TLEs on large)
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int H, W, K, B;
    cin >> H >> W >> K >> B;
    vector<int> S(K), T(K);
    for (int i = 0; i < K; i++) cin >> S[i];
    for (int i = 0; i < K; i++) cin >> T[i];
    vector<vector<int>> blocked_in_row(H + 1);
    for (int i = 0; i < B; i++) {
        int r, c; cin >> r >> c;
        blocked_in_row[r].push_back(c);
    }

    vector<vector<long long>> M(K, vector<long long>(K, 0));
    vector<long long> dp(W + 1, 0);
    vector<char> bad(W + 1, 0);

    for (int i = 0; i < K; i++) {
        fill(dp.begin(), dp.end(), 0);
        dp[S[i]] = 1;
        for (int r = 1; r <= H; r++) {
            for (int c : blocked_in_row[r]) bad[c] = 1;
            for (int c = 1; c <= W; c++) {
                if (bad[c]) {
                    dp[c] = 0;
                } else if (r == 1 && c == S[i]) {
                    dp[c] = 1;
                } else {
                    dp[c] += dp[c - 1];
                    if (dp[c] >= MOD) dp[c] -= MOD;
                }
            }
            for (int c : blocked_in_row[r]) bad[c] = 0;
        }
        for (int j = 0; j < K; j++) M[i][j] = dp[T[j]];
    }

    long long det = 1;
    for (int col = 0; col < K; col++) {
        int piv = -1;
        for (int row = col; row < K; row++) if (M[row][col]) { piv = row; break; }
        if (piv == -1) { det = 0; break; }
        if (piv != col) { swap(M[piv], M[col]); det = (MOD - det) % MOD; }
        det = det * M[col][col] % MOD;
        long long inv_p = modpow(M[col][col], MOD - 2);
        for (int row = col + 1; row < K; row++) if (M[row][col]) {
            long long f = M[row][col] * inv_p % MOD;
            for (int c = col; c < K; c++) {
                M[row][c] = (M[row][c] - f * M[col][c]) % MOD;
                if (M[row][c] < 0) M[row][c] += MOD;
            }
        }
    }
    cout << det << '\n';
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_19())
