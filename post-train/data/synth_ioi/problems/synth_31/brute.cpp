#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<char> on(n + 1, 0);
    vector<int> active;
    long long ans = 0;
    while (q--) {
        int x;
        cin >> x;
        if (!on[x]) {
            for (int y : active) if (std::gcd(a[x], a[y]) == 1) ans++;
            on[x] = 1;
            active.push_back(x);
        } else {
            on[x] = 0;
            active.erase(find(active.begin(), active.end(), x));
            for (int y : active) if (std::gcd(a[x], a[y]) == 1) ans--;
        }
        cout << ans << '\n';
    }
    return 0;
}
