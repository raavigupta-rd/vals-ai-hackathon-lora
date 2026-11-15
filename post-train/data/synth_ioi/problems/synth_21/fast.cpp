#include <bits/stdc++.h>
using namespace std;

template<long long MOD, long long BASE>
struct HashBIT {
    int n;
    vector<long long> pw, ipw, bit;

    static long long modpow(long long a, long long e) {
        long long r = 1;
        while (e > 0) {
            if (e & 1) r = r * a % MOD;
            a = a * a % MOD;
            e >>= 1;
        }
        return r;
    }

    HashBIT(int n) : n(n), pw(n + 2, 1), ipw(n + 2, 1), bit(n + 2, 0) {
        long long ib = modpow(BASE, MOD - 2);
        for (int i = 1; i <= n + 1; i++) {
            pw[i] = pw[i - 1] * BASE % MOD;
            ipw[i] = ipw[i - 1] * ib % MOD;
        }
    }
    void add(int idx, long long val) {
        val = (val % MOD + MOD) % MOD;
        for (; idx <= n; idx += idx & -idx) {
            bit[idx] += val;
            if (bit[idx] >= MOD) bit[idx] -= MOD;
        }
    }
    void set_char(int idx, int old_v, int new_v) {
        long long diff = (new_v - old_v) * pw[idx] % MOD;
        add(idx, diff);
    }
    long long sum(int idx) const {
        long long s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s % MOD;
    }
    long long get_hash(int l, int r) const {
        long long raw = (sum(r) - sum(l - 1) + MOD) % MOD;
        return raw * ipw[l] % MOD;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;
    HashBIT<1000000007LL, 137LL> f1(n), r1(n);
    HashBIT<1000000009LL, 139LL> f2(n), r2(n);

    for (int i = 1; i <= n; i++) {
        int v = s[i - 1] - 'a' + 1;
        f1.set_char(i, 0, v);
        r1.set_char(n - i + 1, 0, v);
        f2.set_char(i, 0, v);
        r2.set_char(n - i + 1, 0, v);
    }

    while (q--) {
        int t;
        cin >> t;
        if (t == 1) {
            int p; char c;
            cin >> p >> c;
            int old_v = s[p - 1] - 'a' + 1;
            int new_v = c - 'a' + 1;
            if (old_v != new_v) {
                s[p - 1] = c;
                f1.set_char(p, old_v, new_v);
                r1.set_char(n - p + 1, old_v, new_v);
                f2.set_char(p, old_v, new_v);
                r2.set_char(n - p + 1, old_v, new_v);
            }
        } else {
            int l, r;
            cin >> l >> r;
            bool ok = (f1.get_hash(l, r) == r1.get_hash(n - r + 1, n - l + 1)) &&
                      (f2.get_hash(l, r) == r2.get_hash(n - r + 1, n - l + 1));
            cout << (ok ? "YES\n" : "NO\n");
        }
    }
    return 0;
}
