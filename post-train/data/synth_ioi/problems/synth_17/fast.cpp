#include <bits/stdc++.h>
using namespace std;

const long long INF = (1LL << 62);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<long long> p(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        p[i] = p[i - 1] + x;
    }
    vector<long long> prv(n + 1, INF), cur(n + 1, INF);
    for (int i = 1; i <= n; i++) prv[i] = p[i] * p[i];

    function<void(int, int, int, int)> dnc = [&](int l, int r, int optl, int optr) {
        if (l > r) return;
        int mid = (l + r) >> 1;
        pair<long long, int> best = {INF, optl};
        int hi = min(mid - 1, optr);
        for (int j = optl; j <= hi; j++) {
            long long d = p[mid] - p[j];
            long long val = prv[j] + d * d;
            if (val < best.first) best = {val, j};
        }
        cur[mid] = best.first;
        dnc(l, mid - 1, optl, best.second);
        dnc(mid + 1, r, best.second, optr);
    };

    for (int step = 2; step <= k; step++) {
        fill(cur.begin(), cur.end(), INF);
        dnc(step, n, step - 1, n - 1);
        prv.swap(cur);
    }
    cout << prv[n] << '\n';
    return 0;
}
