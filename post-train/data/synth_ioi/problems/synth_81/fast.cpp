#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n), w(n), lo(n, 0), hi(n, 0);
    for (int i = 0; i < n; i++) cin >> a[i];
    for (int i = 0; i < n; i++) cin >> w[i];
    for (int i = 1; i < n; i++) cin >> lo[i] >> hi[i];

    using P = pair<long long, long long>; // (raw_coord, slope_weight)
    priority_queue<P> pqL;
    priority_queue<P, vector<P>, greater<P>> pqR;
    long long shL = 0, shR = 0;
    long long min_val = 0;

    pqL.push({a[0], w[0]});
    pqR.push({a[0], w[0]});

    for (int i = 1; i < n; i++) {
        shL += lo[i];
        shR += hi[i];
        long long ai = a[i], wi = w[i];
        long long l_star = pqL.top().first + shL;
        long long r_star = pqR.top().first + shR;

        if (ai < l_star) {
            long long rem = wi;
            while (rem > 0 && !pqL.empty() && pqL.top().first + shL > ai) {
                auto [xr, cnt] = pqL.top(); pqL.pop();
                long long x = xr + shL;
                long long take = min(rem, cnt);
                min_val += take * (x - ai);
                pqR.push({x - shR, take});
                if (cnt > take) pqL.push({xr, cnt - take});
                rem -= take;
            }
            // (wi - rem) units moved L->R (each added 2 at ai to L), plus rem units in middle (1 to L, 1 to R)
            pqL.push({ai - shL, 2 * wi - rem});
            if (rem > 0) pqR.push({ai - shR, rem});
        } else if (ai > r_star) {
            long long rem = wi;
            while (rem > 0 && !pqR.empty() && pqR.top().first + shR < ai) {
                auto [xr, cnt] = pqR.top(); pqR.pop();
                long long x = xr + shR;
                long long take = min(rem, cnt);
                min_val += take * (ai - x);
                pqL.push({x - shL, take});
                if (cnt > take) pqR.push({xr, cnt - take});
                rem -= take;
            }
            pqR.push({ai - shR, 2 * wi - rem});
            if (rem > 0) pqL.push({ai - shL, rem});
        } else {
            pqL.push({ai - shL, wi});
            pqR.push({ai - shR, wi});
        }
    }

    cout << min_val << '\n';
    return 0;
}
