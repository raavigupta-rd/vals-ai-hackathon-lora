#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
    const long long INF = (1LL << 61);
    vector<long long> best(n, INF);
    vector<char> used(n, 0);
    best[0] = 0;
    long long ans = 0;
    for (int it = 0; it < n; it++) {
        int u = -1;
        for (int i = 0; i < n; i++) {
            if (!used[i] && (u == -1 || best[i] < best[u])) u = i;
        }
        used[u] = 1;
        ans += best[u];
        for (int v = 0; v < n; v++) if (!used[v]) {
            long long d = llabs(x[u] - x[v]) + llabs(y[u] - y[v]);
            if (d < best[v]) best[v] = d;
        }
    }
    cout << ans << '\n';
    return 0;
}
