#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    auto ok = [&](long long lim) -> bool {
        int segs = 1, seg_l = 0;
        long long inv = 0;
        for (int i = 0; i < n; i++) {
            long long add = 0;
            for (int j = seg_l; j < i; j++) if (a[j] > a[i]) add++;
            if (inv + add <= lim) {
                inv += add;
            } else {
                segs++;
                if (segs > k) return false;
                seg_l = i;
                inv = 0;
            }
        }
        return true;
    };

    long long lo = 0, hi = 1LL * n * (n - 1) / 2;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (ok(mid)) hi = mid;
        else lo = mid + 1;
    }
    cout << lo << '\n';
    return 0;
}
