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
