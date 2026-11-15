#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, sz;
    DSU(int n) : p(n + 1), sz(n + 1, 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (sz[a] < sz[b]) swap(a, b);
        p[b] = a; sz[a] += sz[b];
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    struct Edge { int u, v, h; };
    vector<Edge> es(m);
    for (int i = 0; i < m; i++) cin >> es[i].u >> es[i].v >> es[i].h;
    struct Qry { int s, t, id; };
    vector<Qry> qs(q);
    for (int i = 0; i < q; i++) { cin >> qs[i].s >> qs[i].t; qs[i].id = i; }

    sort(es.begin(), es.end(), [](const Edge& a, const Edge& b) { return a.h > b.h; });
    sort(qs.begin(), qs.end(), [](const Qry& a, const Qry& b) { return a.t > b.t; });

    DSU dsu(n);
    vector<int> ans(q);
    int p = 0;
    for (const auto& qr : qs) {
        while (p < m && es[p].h > qr.t) {
            dsu.unite(es[p].u, es[p].v);
            p++;
        }
        ans[qr.id] = dsu.sz[dsu.find(qr.s)];
    }
    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
