#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    int mx = 2;
    for (int i = 1; i <= n; i++) { cin >> a[i]; mx = max(mx, a[i]); }
    vector<int> spf(mx + 1, 0), primes;
    for (int i = 2; i <= mx; i++) {
        if (!spf[i]) { spf[i] = i; primes.push_back(i); }
        for (int p : primes) {
            if (1LL * i * p > mx) break;
            spf[i * p] = p;
            if (i % p == 0) break;
        }
    }
    vector<int> seen(mx + 1, 0);
    for (int step = 1; step <= q; step++) {
        int l, r; cin >> l >> r;
        int cnt = 0;
        for (int i = l; i <= r; i++) {
            int x = a[i];
            while (x > 1) {
                int p = spf[x];
                while (x % p == 0) x /= p;
                if (seen[p] != step) { seen[p] = step; cnt++; }
            }
        }
        cout << cnt << '\n';
    }
    return 0;
}
