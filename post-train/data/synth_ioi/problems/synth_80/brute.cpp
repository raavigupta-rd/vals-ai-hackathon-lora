#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    const int M = 2 * n;
    vector<long long> pos(M);
    long long mn = (1LL << 60), mx = 0;
    for (int i = 0; i < n; i++) {
        cin >> pos[2 * i] >> pos[2 * i + 1];
        mn = min({mn, pos[2 * i], pos[2 * i + 1]});
        mx = max({mx, pos[2 * i], pos[2 * i + 1]});
    }

    // O(N^2) implicit-graph Tarjan 2-SAT using O(N) memory
    vector<int> dfn(M), low(M), comp(M), stk;
    vector<char> in_stk(M, 0);
    struct Frame { int u, v_idx, child; bool ret; };
    vector<Frame> call;

    auto ok = [&](long long D) -> bool {
        if (D <= 0) return true;
        fill(dfn.begin(), dfn.end(), 0);
        fill(comp.begin(), comp.end(), -1);
        int timer = 0, scc_id = 0;
        stk.clear();

        for (int s = 0; s < M; s++) {
            if (dfn[s]) continue;
            call.push_back({s, 0, -1, false});
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
                while (f.v_idx < M) {
                    int q = f.v_idx++;
                    if (q == u || llabs(pos[u] - pos[q]) >= D) continue;
                    int v = q ^ 1;
                    if (!dfn[v]) {
                        f.child = v;
                        f.ret = true;
                        call.push_back({v, 0, -1, false});
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

    long long lo = 0, hi = mx - mn;
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;
        if (ok(mid)) lo = mid;
        else hi = mid - 1;
    }
    cout << lo << '\n';
    return 0;
}
