#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<long long> vals(a.begin() + 1, a.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    vector<int> ca(n + 1);
    for (int i = 1; i <= n; i++) {
        ca[i] = (int)(lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin());
    }
    vector<int> seen(vals.size(), 0);
    int timer = 0;

    long long last_ans = 0;
    while (q--) {
        long long x, y, z;
        cin >> x >> y >> z;
        int l = (int)((x + last_ans) % n) + 1;
        int r = (int)((y + last_ans) % n) + 1;
        if (l > r) swap(l, r);
        int len = r - l + 1;
        int k = (int)((z + last_ans) % len) + 1;

        ++timer;
        int cnt = 0;
        long long res = -1;
        for (int i = l; i <= r; i++) {
            int c = ca[i];
            if (seen[c] != timer) {
                seen[c] = timer;
                if (++cnt == k) {
                    res = a[i];
                    break;
                }
            }
        }
        cout << res << '\n';
        last_ans = (res == -1 ? 0 : res);
    }
    return 0;
}
