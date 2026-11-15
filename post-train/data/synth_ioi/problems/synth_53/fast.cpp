#include <bits/stdc++.h>
using namespace std;

const long long INF = (1LL << 60);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    struct Edge { int u, v; long long w; };
    vector<Edge> es(m + 1);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 1; i <= m; i++) {
        cin >> es[i].u >> es[i].v >> es[i].w;
        adj[es[i].u].push_back({es[i].v, i});
        adj[es[i].v].push_back({es[i].u, i});
    }

    auto dijkstra = [&](int src, vector<long long>& d, vector<int>& par_u, vector<int>& par_e) {
        d.assign(n + 1, INF);
        par_u.assign(n + 1, 0);
        par_e.assign(n + 1, 0);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        d[src] = 0;
        pq.push({0, src});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du != d[u]) continue;
            for (auto [v, eid] : adj[u]) {
                long long nd = du + es[eid].w;
                if (nd < d[v]) {
                    d[v] = nd;
                    par_u[v] = u;
                    par_e[v] = eid;
                    pq.push({nd, v});
                }
            }
        }
    };

    vector<long long> d1, dN;
    vector<int> pu1, pe1, puN, peN;
    dijkstra(1, d1, pu1, pe1);
    dijkstra(n, dN, puN, peN);

    // Extract one shortest path P: v_0=1, v_1, ..., v_K=n
    vector<int> path_v, path_e;
    for (int cur = n; cur != 1; cur = pu1[cur]) {
        path_v.push_back(cur);
        path_e.push_back(pe1[cur]);
    }
    path_v.push_back(1);
    reverse(path_v.begin(), path_v.end());
    reverse(path_e.begin(), path_e.end());
    int K = (int)path_e.size();

    vector<int> pos_v(n + 1, -1), pos_e(m + 1, -1);
    for (int i = 0; i <= K; i++) pos_v[path_v[i]] = i;
    for (int i = 0; i < K; i++) pos_e[path_e[i]] = i; // 0..K-1

    // L_att[u]: earliest index on P of a shortest path from 1 to u (follow P first)
    vector<int> order1(n), orderN(n);
    iota(order1.begin(), order1.end(), 1);
    sort(order1.begin(), order1.end(), [&](int a, int b) { return d1[a] < d1[b]; });
    vector<int> L_att(n + 1, K + 1);
    for (int i = 0; i <= K; i++) L_att[path_v[i]] = i;
    for (int u : order1) if (d1[u] < INF) {
        for (auto [v, eid] : adj[u]) {
            if (pos_e[eid] == -1 && d1[u] + es[eid].w == d1[v]) {
                L_att[v] = min(L_att[v], L_att[u]);
            }
        }
    }

    iota(orderN.begin(), orderN.end(), 1);
    sort(orderN.begin(), orderN.end(), [&](int a, int b) { return dN[a] < dN[b]; });
    vector<int> R_att(n + 1, -1);
    for (int i = 0; i <= K; i++) R_att[path_v[i]] = i;
    for (int u : orderN) if (dN[u] < INF) {
        for (auto [v, eid] : adj[u]) {
            if (pos_e[eid] == -1 && dN[u] + es[eid].w == dN[v]) {
                R_att[v] = max(R_att[v], R_att[u]);
            }
        }
    }

    vector<vector<pair<long long, int>>> add_at(K + 1);
    for (int i = 1; i <= m; i++) {
        if (pos_e[i] != -1) continue;
        int u = es[i].u, v = es[i].v;
        long long w = es[i].w;
        for (int dir = 0; dir < 2; dir++) {
            if (d1[u] < INF && dN[v] < INF && L_att[u] < R_att[v]) {
                add_at[L_att[u]].push_back({d1[u] + w + dN[v], R_att[v]});
            }
            swap(u, v);
        }
    }

    vector<long long> repl(K, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    for (int i = 0; i < K; i++) {
        for (auto& pr : add_at[i]) pq.push(pr);
        while (!pq.empty() && pq.top().second <= i) pq.pop();
        if (!pq.empty()) repl[i] = pq.top().first;
    }

    long long base = d1[n];
    while (q--) {
        int eid; cin >> eid;
        int idx = pos_e[eid];
        long long ans = (idx == -1 ? base : repl[idx]);
        cout << (ans >= INF ? -1LL : ans) << '\n';
    }
    return 0;
}
