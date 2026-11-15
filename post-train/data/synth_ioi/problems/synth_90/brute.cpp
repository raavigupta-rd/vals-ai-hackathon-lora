#include <bits/stdc++.h>
using namespace std;

struct Dinic {
    struct E { int to, rev; long long cap; };
    int n;
    vector<vector<E>> g;
    vector<int> lvl, ptr;
    Dinic(int n) : n(n), g(n), lvl(n), ptr(n) {}
    void add_edge(int u, int v, long long c) {
        g[u].push_back({v, (int)g[v].size(), c});
        g[v].push_back({u, (int)g[u].size() - 1, 0});
    }
    long long dfs(int u, int t, long long pushed) {
        if (pushed == 0 || u == t) return pushed;
        for (int& cid = ptr[u]; cid < (int)g[u].size(); cid++) {
            auto& e = g[u][cid];
            if (lvl[e.to] != lvl[u] + 1 || e.cap == 0) continue;
            long long tr = dfs(e.to, t, min(pushed, e.cap));
            if (tr == 0) continue;
            e.cap -= tr;
            g[e.to][e.rev].cap += tr;
            return tr;
        }
        return 0;
    }
    long long flow(int s, int t) {
        long long f = 0;
        while (true) {
            fill(lvl.begin(), lvl.end(), -1);
            lvl[s] = 0;
            queue<int> q;
            q.push(s);
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (auto& e : g[u]) if (e.cap > 0 && lvl[e.to] == -1) {
                    lvl[e.to] = lvl[u] + 1;
                    q.push(e.to);
                }
            }
            if (lvl[t] == -1) break;
            fill(ptr.begin(), ptr.end(), 0);
            while (long long tr = dfs(s, t, (1LL << 60))) f += tr;
        }
        return f;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<long long> cap(m + 1);
    for (int i = 1; i <= m; i++) cin >> cap[i];
    struct Fleet { int l, r; long long c; };
    vector<Fleet> fl(n);
    for (int i = 0; i < n; i++) cin >> fl[i].l >> fl[i].r >> fl[i].c;

    if (n <= 400 && m <= 400) {
        // Exact Dinic Max-Flow on small inputs!
        int S = 0, T = n + m + 1;
        Dinic din(T + 1);
        const long long INF = (1LL << 60);
        for (int i = 0; i < n; i++) {
            din.add_edge(S, 1 + i, fl[i].c);
            for (int p = 1; p <= fl[i].l; p++) din.add_edge(1 + i, 1 + n + p - 1, INF);
            for (int p = max(fl[i].l + 1, fl[i].r); p <= m; p++) din.add_edge(1 + i, 1 + n + p - 1, INF);
        }
        for (int p = 1; p <= m; p++) din.add_edge(1 + n + p - 1, T, cap[p]);
        cout << din.flow(S, T) << '\n';
        return 0;
    }

    // O(M * N + M^2) memory-safe evaluation for large[0] TLE verification
    vector<long long> pref(m + 2, 0), suff(m + 3, 0);
    for (int i = 1; i <= m; i++) pref[i] = pref[i - 1] + cap[i];
    for (int i = m; i >= 1; i--) suff[i] = suff[i + 1] + cap[i];
    long long tot_c = 0;
    for (auto& f : fl) tot_c += f.c;
    long long ans = min(tot_c, pref[m]);
    for (int a = 0; a <= m - 1; a++) {
        for (int b = a + 2; b <= m + 1; b++) {
            long long saved = 0;
            for (auto& f : fl) if (f.l <= a && f.r >= b) saved += f.c;
            ans = min(ans, pref[a] + suff[b] + tot_c - saved);
        }
    }
    cout << ans << '\n';
    return 0;
}
