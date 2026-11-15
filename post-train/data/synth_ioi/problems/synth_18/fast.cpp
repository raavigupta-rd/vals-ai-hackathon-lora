#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int B = 20;
    const int SZ = 1 << B;
    vector<long long> f(SZ, 0);
    for (int i = 0; i < n; i++) {
        int m; long long p;
        cin >> m >> p;
        f[m] += p;
    }
    for (int b = 0; b < B; b++) {
        int bit = 1 << b;
        for (int mask = 0; mask < SZ; mask++) {
            if (mask & bit) f[mask] += f[mask ^ bit];
        }
    }
    while (q--) {
        int x;
        cin >> x;
        cout << f[x] << '\n';
    }
    return 0;
}
