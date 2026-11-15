#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> p(n + 1);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> p[i];
    for (int i = 1; i <= n; i++) cin >> w[i];
    while (q--) {
        int u; long long k;
        cin >> u >> k;
        long long total = 0;
        for (long long step = 0; step < k; step++) {
            total += w[u];
            u = p[u];
        }
        cout << u << ' ' << total << '\n';
    }
    return 0;
}
