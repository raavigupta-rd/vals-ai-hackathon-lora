#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k, L;
    cin >> n >> k >> L;
    vector<long long> pref(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        pref[i] = pref[i - 1] + x;
    }
    k = min(k, n / L);
    vector<long long> prv(n + 1, 0), cur(n + 1, 0);
    for (int j = 1; j <= k; j++) {
        fill(cur.begin(), cur.begin() + L, 0);
        for (int i = L; i <= n; i++) {
            cur[i] = max(cur[i - 1], prv[i - L] + (pref[i] - pref[i - L]));
        }
        swap(prv, cur);
    }
    cout << prv[n] << '\n';
    return 0;
}
