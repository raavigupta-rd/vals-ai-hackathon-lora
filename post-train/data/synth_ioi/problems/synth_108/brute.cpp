#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<int> pos_in_act(n + 1, -1), act;
    long long ans = 0;
    while (q--) {
        int x;
        cin >> x;
        long long vx = a[x];
        if (pos_in_act[x] == -1) {
            for (int y : act) {
                long long vy = a[y];
                long long l = (vx / std::gcd(vx, vy)) % MOD * (vy % MOD) % MOD;
                ans += l;
                if (ans >= MOD) ans -= MOD;
            }
            pos_in_act[x] = (int)act.size();
            act.push_back(x);
        } else {
            int idx = pos_in_act[x];
            int last = act.back();
            act[idx] = last;
            pos_in_act[last] = idx;
            act.pop_back();
            pos_in_act[x] = -1;
            for (int y : act) {
                long long vy = a[y];
                long long l = (vx / std::gcd(vx, vy)) % MOD * (vy % MOD) % MOD;
                ans -= l;
                if (ans < 0) ans += MOD;
            }
        }
        cout << ans << '\n';
    }
    return 0;
}
