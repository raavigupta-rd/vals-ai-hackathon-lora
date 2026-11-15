#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, C, q;
    cin >> n >> C >> q;
    vector<int> w(n + 1);
    vector<long long> v(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i] >> v[i];

    int stride = C + 1;
    vector<long long> pref((n + 2) * stride, 0), suff((n + 2) * stride, 0);

    for (int i = 1; i <= n; i++) {
        const long long* prv = &pref[(i - 1) * stride];
        long long* cur = &pref[i * stride];
        int wi = w[i];
        long long vi = v[i];
        memcpy(cur, prv, sizeof(long long) * stride);
        for (int c = wi; c <= C; c++) {
            cur[c] = max(cur[c], prv[c - wi] + vi);
        }
    }
    for (int i = n; i >= 1; i--) {
        const long long* nxt = &suff[(i + 1) * stride];
        long long* cur = &suff[i * stride];
        int wi = w[i];
        long long vi = v[i];
        memcpy(cur, nxt, sizeof(long long) * stride);
        for (int c = wi; c <= C; c++) {
            cur[c] = max(cur[c], nxt[c - wi] + vi);
        }
    }

    while (q--) {
        int k, b;
        cin >> k >> b;
        const long long* L = &pref[(k - 1) * stride];
        const long long* R = &suff[(k + 1) * stride];
        long long best = 0;
        for (int x = 0; x <= b; x++) {
            best = max(best, L[x] + R[b - x]);
        }
        cout << best << '\n';
    }
    return 0;
}
