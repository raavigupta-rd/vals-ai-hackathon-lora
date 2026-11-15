#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int d;
    long long k;
    cin >> d >> k;
    vector<int> a(d), c(d);
    for (int i = 0; i < d; i++) cin >> a[i];
    for (int j = 0; j < d; j++) cin >> c[j];
    if (k < d) {
        cout << a[k] % MOD << '\n';
        return 0;
    }
    // Rolling ring buffer O(D) memory, O(K * D) time
    vector<int> buf = a;
    int head = 0;
    for (long long step = d; step <= k; step++) {
        long long acc = 0;
        for (int j = 1; j <= d; j++) {
            int idx = head - j;
            if (idx < 0) idx += d;
            acc += (long long)c[j - 1] * buf[idx] % MOD;
        }
        buf[head] = (int)(acc % MOD);
        head++;
        if (head == d) head = 0;
    }
    int last = (head == 0 ? d - 1 : head - 1);
    cout << buf[last] << '\n';
    return 0;
}
