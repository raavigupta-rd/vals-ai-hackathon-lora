#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    int mx = 1;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        mx = max(mx, a[i]);
    }
    vector<int> spf(mx + 1, 0), mu(mx + 1, 0);
    mu[1] = 1;
    vector<int> primes;
    for (int i = 2; i <= mx; i++) {
        if (!spf[i]) { spf[i] = i; primes.push_back(i); mu[i] = -1; }
        for (int p : primes) {
            if (1LL * i * p > mx) break;
            spf[i * p] = p;
            if (i % p == 0) { mu[i * p] = 0; break; }
            mu[i * p] = -mu[i];
        }
    }

    auto sqfree_divs = [&](int x, vector<int>& divs) {
        int ps[10], k = 0;
        while (x > 1) {
            int p = spf[x];
            ps[k++] = p;
            while (x % p == 0) x /= p;
        }
        divs.resize(1 << k);
        divs[0] = 1;
        for (int b = 0; b < k; b++) {
            int sz = 1 << b, p = ps[b];
            for (int m = 0; m < sz; m++) divs[sz + m] = divs[m] * p;
        }
    };

    vector<int> cnt(mx + 1, 0);
    vector<char> on(n + 1, 0);
    vector<int> divs;
    long long ans = 0;
    while (q--) {
        int idx;
        cin >> idx;
        sqfree_divs(a[idx], divs);
        if (!on[idx]) {
            on[idx] = 1;
            for (int d : divs) {
                ans += 1LL * mu[d] * cnt[d];
                cnt[d]++;
            }
        } else {
            on[idx] = 0;
            for (int d : divs) {
                cnt[d]--;
                ans -= 1LL * mu[d] * cnt[d];
            }
        }
        cout << ans << '\n';
    }
    return 0;
}
