#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<long long, long long>>> sets(n);
    for (int i = 0; i < n; i++) {
        int m;
        cin >> m;
        sets[i].resize(m);
        for (int j = 0; j < m; j++) cin >> sets[i][j].first >> sets[i][j].second;
    }
    while (q--) {
        long long A, B;
        cin >> A >> B;
        long long total = 0;
        for (int i = 0; i < n; i++) {
            long long best = -(1LL << 62);
            for (auto [x, y] : sets[i]) {
                best = max(best, A * x + B * y);
            }
            total += best;
        }
        cout << total << '\n';
    }
    return 0;
}
