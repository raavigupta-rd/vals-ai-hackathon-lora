#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long K;
    cin >> n >> K;
    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    sort(a.begin(), a.end());

    vector<long long> pref(n + 1, 0);
    for (int i = 0; i < n; i++) pref[i + 1] = pref[i] + a[i];

    auto count_le = [&](long long D) -> long long {
        long long cnt = 0;
        int l = 0;
        for (int r = 0; r < n; r++) {
            while (a[r] - a[l] > D) l++;
            cnt += (r - l);
        }
        return cnt;
    };

    long long lo = 0, hi = a[n - 1] - a[0];
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (count_le(mid) >= K) hi = mid;
        else lo = mid + 1;
    }
    long long dK = lo;

    long long cnt_lt = 0, sum_lt = 0;
    if (dK > 0) {
        int l = 0;
        for (int r = 0; r < n; r++) {
            while (a[r] - a[l] >= dK) l++;
            long long c = r - l;
            cnt_lt += c;
            sum_lt += c * a[r] - (pref[r] - pref[l]);
        }
    }
    long long sK = sum_lt + (K - cnt_lt) * dK;
    cout << dK << ' ' << sK << '\n';
    return 0;
}
