#include <bits/stdc++.h>
using namespace std;

struct Line {
    long long m, b;
    bool empty = true;
    inline __int128 eval(long long x) const {
        if (empty) return (__int128)1 << 120;
        return (__int128)m * x + b;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<long long> S(n + 1, 0), P(n + 1, 0), xs;
    xs.reserve(n);
    for (int u = 2; u <= n; u++) {
        cin >> S[u] >> P[u];
        xs.push_back(P[u]);
    }
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());
    int M = (int)xs.size();

    vector<Line> seg(4 * M + 5);
    vector<pair<int, Line>> undo;
    undo.reserve(n * 20);

    auto insert_line = [&](Line nw) {
        int u = 1, l = 0, r = M - 1;
        while (true) {
            int mid = (l + r) >> 1;
            bool left_better = nw.eval(xs[l]) < seg[u].eval(xs[l]);
            bool mid_better = nw.eval(xs[mid]) < seg[u].eval(xs[mid]);
            if (mid_better) {
                undo.push_back({u, seg[u]});
                swap(seg[u], nw);
            }
            if (l == r || nw.empty) break;
            if (left_better != mid_better) {
                u = u << 1;
                r = mid;
            } else {
                u = u << 1 | 1;
                l = mid + 1;
            }
        }
    };

    auto query_x = [&](int idx) -> long long {
        long long x = xs[idx];
        __int128 res = (__int128)1 << 120;
        int u = 1, l = 0, r = M - 1;
        while (true) {
            res = min(res, seg[u].eval(x));
            if (l == r) break;
            int mid = (l + r) >> 1;
            if (idx <= mid) { u = u << 1; r = mid; }
            else { u = u << 1 | 1; l = mid + 1; }
        }
        return (long long)res;
    };

    vector<long long> dist(n + 1, 0), dp(n + 1, 0);
    vector<int> par(n + 1, 0);
    struct Frame { int u, idx; size_t snap; };
    vector<Frame> st = {{1, 0, 0}};

    while (!st.empty()) {
        auto& f = st.back();
        int u = f.u;
        if (f.idx == 0) {
            f.snap = undo.size();
            if (u != 1) {
                int qi = (int)(lower_bound(xs.begin(), xs.end(), P[u]) - xs.begin());
                dp[u] = S[u] + dist[u] * P[u] + query_x(qi);
            }
            insert_line({-dist[u], dp[u], false});
        }
        bool pushed = false;
        while (f.idx < (int)adj[u].size()) {
            auto [v, w] = adj[u][f.idx++];
            if (v == par[u]) continue;
            par[v] = u;
            dist[v] = dist[u] + w;
            st.push_back({v, 0, 0});
            pushed = true;
            break;
        }
        if (!pushed) {
            while (undo.size() > f.snap) {
                auto [node, old_l] = undo.back();
                undo.pop_back();
                seg[node] = old_l;
            }
            st.pop_back();
        }
    }

    for (int u = 2; u <= n; u++) {
        cout << dp[u] << (u == n ? '\n' : ' ');
    }
    return 0;
}
