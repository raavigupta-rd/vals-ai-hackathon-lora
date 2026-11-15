#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int LOG = 51; // 2^50 > 10^15
    vector<array<int, LOG>> up(n + 1);
    vector<array<long long, LOG>> sm(n + 1);
    for (int i = 1; i <= n; i++) cin >> up[i][0];
    for (int i = 1; i <= n; i++) cin >> sm[i][0];

    for (int b = 1; b < LOG; b++) {
        for (int i = 1; i <= n; i++) {
            int mid = up[i][b - 1];
            up[i][b] = up[mid][b - 1];
            sm[i][b] = sm[i][b - 1] + sm[mid][b - 1];
        }
    }
    while (q--) {
        int u; long long k;
        cin >> u >> k;
        long long total = 0;
        for (int b = 0; b < LOG; b++) if ((k >> b) & 1LL) {
            total += sm[u][b];
            u = up[u][b];
        }
        cout << u << ' ' << total << '\n';
    }
    return 0;
}
