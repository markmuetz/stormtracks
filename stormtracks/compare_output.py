import sys
import gc
import datetime as dt

from results import StormtracksResultsManager
from load_settings import settings

YEARS = range(1890, 2010)
NUM_ENSEMBLE_MEMBERS = 56

KNOWN_BAD = (
    (1890, 27),
)

def flush_print(msg):
    print(msg)
    sys.stdout.flush()


def decompress_all():
    from load_settings import settings
    srm = StormtracksResultsManager('prod_release_1')
    for year in YEARS:
        flush_print('Decompressing year: {0}'.format(year))
        try:
            srm.decompress_year(year)
        except Exception as e:
            flush_print('Failed to decompress {0}: {1}'.format(year, e))


def output1(year):
    srm = StormtracksResultsManager('prod_release_1')
    return get_all_results('aws', srm, srm, year)


def output2(year):
    output_dir = settings.SECOND_OUTPUT_DIR

    tracking_srm = StormtracksResultsManager('pyro_tracking_analysis', output_dir)
    field_col_srm = StormtracksResultsManager('pyro_field_collection_analysis', output_dir)

    return get_all_results('ucl', tracking_srm, field_col_srm, year)


def get_all_results(name, tracking_srm, field_col_srm, year):
    vt_key = 'vort_tracks_by_date-scale:3;pl:850;tracker:nearest_neighbour'
    gm_key = 'good_matches-scale:3;pl:850;tracker:nearest_neighbour'
    cs_key = 'cyclones'

    all_vort_tracks, all_good_matches, all_cyclones = [], [], []
    skip_members = set()
    for i in range(NUM_ENSEMBLE_MEMBERS):
        flush_print('loading {0} results for year {1} ({2}/{3})'.format(name, year, 
                                                                  i + 1, NUM_ENSEMBLE_MEMBERS))
        try:
            if (year, i) in KNOWN_BAD:
                raise Exception('Known to be bad: {0}'.format((year, i)))
            vort_tracks = tracking_srm.get_result(year, i, vt_key)
            good_matches = tracking_srm.get_result(year, i, gm_key)
            cyclones = field_col_srm.get_result(year, i, cs_key)

            all_vort_tracks.append(vort_tracks)
            all_good_matches.append(good_matches)
            all_cyclones.append(cyclones)
        except Exception as e:
            flush_print('SKIPPING {0} results for year {1} ({2}/{3})'.format(name, year, 
                                                                       i + 1, NUM_ENSEMBLE_MEMBERS))
            flush_print(e)
            all_vort_tracks.append([])
            all_good_matches.append([])
            all_cyclones.append([])
            skip_members.add(i)

    return all_vort_tracks, all_good_matches, all_cyclones, skip_members


def compare_all_vort_tracks(all_vort_tracks1, all_vort_tracks2, skip_members):
    total1, total2 = 0, 0
    flush_print('Compare vort tracks')
    for total, all_vort_tracks in [(total1, all_vort_tracks1), 
                                   (total2, all_vort_tracks2)]:
        for i in range(NUM_ENSEMBLE_MEMBERS):
            if i in skip_members:
                flush_print('Skipping member {0}'.format(i))
                continue
            vort_tracks_by_date = all_vort_tracks[i]
            vort_tracks_for_dates = vort_tracks_by_date.values()
            for vort_tracks in vort_tracks_for_dates:
                for vort_track in vort_tracks:
                    for vortmax in vort_track.vortmaxes:
                        total += vortmax.vort
        flush_print(total)
    flush_print('  Diff in totals: {0}'.format(abs(total2 - total1)))
    if abs(total2 - total1) > 1e-5:
        raise Exception('VORT_TRACKS: Totals of vortmax are too far apart!')
    else:
        flush_print('  vort tracks are acceptably similar')


def compare_all_good_matches(all_good_matches1, all_good_matches2, skip_members):
    total1, total2 = 0, 0
    for total, all_good_matches in [(total1, all_good_matches1), 
                                    (total2, all_good_matches2)]:
        for i in range(NUM_ENSEMBLE_MEMBERS):
            if i in skip_members:
                flush_print('Skipping member {0}'.format(i))
                continue
            gms = all_good_matches[i]
            for m in gms:
                total += m.av_dist()
        flush_print(total)
    flush_print('  Diff in totals: {0}'.format(abs(total2 - total1)))
    if abs(total2 - total1) > 1e-5:
        raise Exception('GOOD_MATCHES: Totals of av_dist are too far apart!')
    else:
        flush_print('  good matches are acceptably similar')


def compare_all_cyclones(all_cyclones1, all_cyclones2, skip_members):
    total1, total2 = 0, 0
    for total, all_cyclones, get_vals in [(total1, all_cyclones1, True), 
                                          (total2, all_cyclones2, False)]:
        for i in range(NUM_ENSEMBLE_MEMBERS):
            if i in skip_members:
                flush_print('Skipping member {0}'.format(i))
                continue
            if get_vals:
                cyclones = all_cyclones[i].values()
            else:
                cyclones = all_cyclones[i]
            for ct in cyclones:
                total += sum(ct.t995s.values())
                total += sum(ct.capes.values())

        flush_print(total)
    flush_print('  Diff in totals: {0}'.format(abs(total2 - total1)))
    if abs(total2 - total1) > 1e-5:
        raise Exception('CYCLONES: Totals of av_dist are too far apart!')
    else:
        flush_print('  cyclones are acceptably similar')


def main():
    flush_print('Start: {0}'.format(dt.datetime.now()))
    
    for year in YEARS:
        all_vort_tracks1, all_good_matches1, all_cyclones1, skip_members1 = output1(year)
        all_vort_tracks2, all_good_matches2, all_cyclones2, skip_members2 = output2(year)
        skip_members = skip_members1 | skip_members2

        compare_all_vort_tracks(all_vort_tracks1, all_vort_tracks2, skip_members)
        compare_all_good_matches(all_good_matches1, all_good_matches2, skip_members)
        compare_all_cyclones(all_cyclones1, all_cyclones2, skip_members)

        del all_vort_tracks1
        del all_good_matches1
        del all_cyclones1

        del all_vort_tracks2
        del all_good_matches2
        del all_cyclones2

        unreachable_objs = gc.collect()
        flush_print('# unreachable: {0}'.format(unreachable_objs))
    flush_print('End: {0}'.format(dt.datetime.now()))


if __name__ == '__main__':
    main()
