#include <bits/stdc++.h>
using namespace std;

static inline int sgn(__int128 v) { return (v > 0) - (v < 0); }

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
    auto cr = [&](int i, int j) -> __int128 {
        return (__int128)x[i] * y[j] - (__int128)y[i] * x[j];
    };
    long long ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int s1 = sgn(cr(i, j));
            for (int k = j + 1; k < n; k++) {
                int s2 = sgn(cr(j, k));
                int s3 = sgn(cr(k, i));
                if (s1 == s2 && s2 == s3 && s1 != 0) ans++;
            }
        }
    }
    cout << ans << '\n';
    return 0;
}
