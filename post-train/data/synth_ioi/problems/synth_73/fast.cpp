#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    vector<int> vals = a;
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    vector<int> rk(n);
    for (int i = 0; i < n; i++) {
        rk[i] = (int)(lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin()) + 1;
    }

    vector<int> bit(m + 1, 0);
    auto bit_add = [&](int idx, int d) {
        for (; idx <= m; idx += idx & -idx) bit[idx] += d;
    };
    auto bit_sum = [&](int idx) {
        int s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };

    auto ok = [&](long long lim) -> bool {
        int segs = 1;
        int seg_l = 0;
        long long inv = 0;
        for (int i = 0; i < n; i++) {
            int len = i - seg_l;
            int le = bit_sum(rk[i]);
            long long add = len - le;
            if (inv + add <= lim) {
                inv += add;
                bit_add(rk[i], +1);
            } else {
                for (int j = seg_l; j < i; j++) bit_add(rk[j], -1);
                segs++;
                if (segs > k) return false;
                seg_l = i;
                inv = 0;
                bit_add(rk[i], +1);
            }
        }
        for (int j = seg_l; j < n; j++) bit_add(rk[j], -1);
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
