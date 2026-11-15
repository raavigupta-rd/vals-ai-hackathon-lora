#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

struct Pt {
    int x, y, z;
    long long w;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<Pt> p(n);
    for (int i = 0; i < n; i++) cin >> p[i].x >> p[i].y >> p[i].z >> p[i].w;
    sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    });
    vector<long long> best_v(n);
    vector<int> ways(n);
    long long ans_v = 0;
    int ans_w = 0;
    for (int i = 0; i < n; i++) {
        long long pv = 0;
        int pw = 0;
        for (int j = 0; j < i; j++) {
            if (p[j].x < p[i].x && p[j].y < p[i].y && p[j].z < p[i].z) {
                if (best_v[j] > pv) {
                    pv = best_v[j];
                    pw = ways[j];
                } else if (best_v[j] == pv) {
                    pw += ways[j];
                    if (pw >= MOD) pw -= MOD;
                }
            }
        }
        if (pw == 0) {
            best_v[i] = p[i].w;
            ways[i] = 1;
        } else {
            best_v[i] = pv + p[i].w;
            ways[i] = pw;
        }
        if (best_v[i] > ans_v) {
            ans_v = best_v[i];
            ans_w = ways[i];
        } else if (best_v[i] == ans_v) {
            ans_w = (ans_w + ways[i]) % MOD;
        }
    }
    cout << ans_v << ' ' << ans_w << '\n';
    return 0;
}
