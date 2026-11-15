#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long K;
    cin >> n >> K;
    vector<long long> a(n);
    long long mn = (1LL << 60), mx = 0;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        mn = min(mn, a[i]);
        mx = max(mx, a[i]);
    }

    auto count_le = [&](long long D) -> long long {
        long long cnt = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (llabs(a[i] - a[j]) <= D) cnt++;
            }
        }
        return cnt;
    };

    long long lo = 0, hi = mx - mn;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (count_le(mid) >= K) hi = mid;
        else lo = mid + 1;
    }
    long long dK = lo, cnt_lt = 0, sum_lt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            long long d = llabs(a[i] - a[j]);
            if (d < dK) {
                cnt_lt++;
                sum_lt += d;
            }
        }
    }
    cout << dK << ' ' << (sum_lt + (K - cnt_lt) * dK) << '\n';
    return 0;
}
