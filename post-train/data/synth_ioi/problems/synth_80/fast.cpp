#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    const int M = 2 * n;
    struct Slot {
        long long x;
        int lit;
        bool operator<(const Slot& o) const { return x < o.x; }
    };
    vector<Slot> slots(M);
    for (int i = 0; i < n; i++) {
        long long a, b;
        cin >> a >> b;
        slots[2 * i] = {a, 2 * i};
        slots[2 * i + 1] = {b, 2 * i + 1};
    }
    sort(slots.begin(), slots.end());
    vector<long long> xs(M);
    for (int i = 0; i < M; i++) xs[i] = slots[i].x;

    // Graph layout:
    // 0 .. M-1: real literals (2*i and 2*i+1)
    // M + (1 .. 2*M - 1): iterative segment-tree nodes where leaf M + (M + p) -> (slots[p].lit ^ 1)
    const int V = M + 2 * M;
    const int MAXE = 2 * M + M * 36 + 100;
    vector<int> head(V, -1), to(MAXE), nxt(MAXE);
    int ecnt = 0;
    auto add_edge = [&](int u, int v) {
        to[ecnt] = v;
        nxt[ecnt] = head[u];
        head[u] = ecnt++;
    };

    // Prebuild static segment-tree edges once; snapshot head_base and ecnt_base!
    for (int u = 1; u < M; u++) {
        add_edge(M + u, M + 2 * u);
        add_edge(M + u, M + 2 * u + 1);
    }
    for (int p = 0; p < M; p++) {
        add_edge(M + (M + p), slots[p].lit ^ 1);
    }
    vector<int> head_base = head;
    const int ecnt_base = ecnt;

    vector<int> dfn(V, 0), low(V, 0), comp(V, -1), stk;
    vector<char> in_stk(V, 0);
    stk.reserve(V);
    struct Frame { int u, e, child; bool ret; };
    vector<Frame> call;
    call.reserve(V);

    auto add_range = [&](int src, int l, int r) {
        for (l += M, r += M + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) add_edge(src, M + (l++));
            if (r & 1) add_edge(src, M + (--r));
        }
    };

    auto ok = [&](long long D) -> bool {
        if (D <= 0) return true;
        // Only real literals 0..M-1 got extra edges; restore them in O(M)!
        for (int i = 0; i < M; i++) head[i] = head_base[i];
        ecnt = ecnt_base;

        int L = 0, R = 0;
        for (int p = 0; p < M; p++) {
            while (xs[p] - xs[L] >= D) L++;
            while (R + 1 < M && xs[R + 1] - xs[p] < D) R++;
            int lit = slots[p].lit;
            if (L <= p - 1) add_range(lit, L, p - 1);
            if (p + 1 <= R) add_range(lit, p + 1, R);
        }

        fill(dfn.begin(), dfn.end(), 0);
        fill(comp.begin(), comp.end(), -1);
        int timer = 0, scc_id = 0;
        stk.clear();

        for (int s = 0; s < M; s++) {
            if (dfn[s]) continue;
            call.push_back({s, head[s], -1, false});
            while (!call.empty()) {
                auto& f = call.back();
                int u = f.u;
                if (!dfn[u]) {
                    dfn[u] = low[u] = ++timer;
                    stk.push_back(u);
                    in_stk[u] = 1;
                }
                if (f.ret) {
                    low[u] = min(low[u], low[f.child]);
                    f.ret = false;
                }
                bool pushed = false;
                while (f.e != -1) {
                    int v = to[f.e];
                    f.e = nxt[f.e];
                    if (!dfn[v]) {
                        f.child = v;
                        f.ret = true;
                        call.push_back({v, head[v], -1, false});
                        pushed = true;
                        break;
                    } else if (in_stk[v]) {
                        low[u] = min(low[u], dfn[v]);
                    }
                }
                if (!pushed) {
                    if (low[u] == dfn[u]) {
                        while (true) {
                            int x = stk.back(); stk.pop_back();
                            in_stk[x] = 0;
                            comp[x] = scc_id;
                            if (x == u) break;
                        }
                        scc_id++;
                    }
                    call.pop_back();
                }
            }
        }

        for (int i = 0; i < n; i++) {
            if (comp[2 * i] == comp[2 * i + 1]) return false;
        }
        return true;
    };

    long long lo = 0, hi = xs[M - 1] - xs[0];
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;
        if (ok(mid)) lo = mid;
        else hi = mid - 1;
    }
    cout << lo << '\n';
    return 0;
}
