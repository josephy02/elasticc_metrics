import argparse
import logging
import os
from pprint import pformat
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests


TAXONOMY = {
    -1: 'missed',
    0: 'Static/Other',
    1: 'Non-Recurring',
    2: 'Recurring',
    20: 'Recurring/Other',
    21: 'Periodic',
    22: 'Non-Periodic',
    210: 'Periodic/Other',
    211: 'Cepheid',
    212: 'RR Lyrae',
    213: 'Delta Scuti',
    214: 'EB',
    215: 'LPV/Mira',
    220: 'Non-Periodic/Other',
    221: 'AGN',
    10: 'Non-Recurring/Other',
    11: 'SN-like',
    12: 'Fast',
    13: 'Long',
    110: 'SN-like/Other',
    111: 'Ia',
    112: 'Ib/c',
    113: 'II',
    114: 'Iax',
    115: '91bg',
    120: 'Fast/Other',
    121: 'KN',
    122: 'M-dwarf Flare',
    123: 'Dwarf Novae',
    124: 'uLens',
    130: 'Long/Other',
    131: 'SLSN',
    132: 'TDE',
    133: 'ILOT',
    134: 'CART',
    135: 'PISN',
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog='Confusion matrices for the last broker classifications',
        description='Get confusion matrices for true and last predicted classes for each pair of diaObject and classifierId',
    )
    parser.add_argument('--include-missed', action='store_true', help='Add missed classifications as a predicted class')
    parser.add_argument('--plot', action='store_true', help='Plot and save confusion matrices')
    parser.add_argument('--save', action='store_true', help='Save CSV with data')
    parser.add_argument('--norm', default='true', choices=['true', 'pred', 'all'],
                        help='how t onormalize confusion matrices')
    parser.add_argument('--definition', default='last_best', choices=['last_best', 'best'],
                        help='''definition of classification:
                            "best" means having the largest probability over all classifier messages,
                            "last_best" means having the largest probability for the most recent alert
                        ''')
    parser.add_argument('--classifier_id', type=int, help='consider a single classifier')
    return parser.parse_args(args)


class Client:
    url = "https://desc-tom.lbl.gov"

    def __init__(self, session: requests.Session):
        self.session = session

    @classmethod
    def from_credentials(cls, user, password):
        session = requests.session()
        session.get(f'{cls.url}/accounts/login/')
        res = session.post(
            f'{cls.url}/accounts/login/',
            data={
                "username": user,
                "password": password,
                "csrfmiddlewaretoken": session.cookies['csrftoken']
            },
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to log in; http status: {res.status_code}")
        if 'Please enter a correct' in res.text:
            # This is a very cheesy attempt at checking if the login failed.
            # I haven't found clean documentation on how to log into a django site
            # from an app like this using standard authentication stuff.  So, for
            # now, I'm counting on the HTML that happened to come back when
            # I ran it with a failed login one time.  One of these days I'll actually
            # figure out how Django auth works and make a version of /accounts/login/
            # designed for use in API scripts like this one, rather than desgined
            # for interactive users.
            raise RuntimeError("Failed to log in.  I think.  Put in a debug break and look at res.text")
        session.headers['X-CSRFToken'] = session.cookies['csrftoken']
        return cls(session)

    def __call__(self, query: str) -> List[Dict]:
        result = self.session.post(f'{self.url}/db/runsqlquery/', json={'query': query, 'subdict': {}})
        result.raise_for_status()
        data = result.json()
        if ('status' not in data) or (data['status'] != 'ok'):
            raise RuntimeError(f"Got unexpected response:\n{data}\n")
        return data['rows']


def get_classifications(*,
                        definition: str,
                        classifier_id: Optional[int],
                        include_missed: bool = False) -> Dict[str, pd.DataFrame]:
    username = os.getenv("DESC_TOM_USERNAME", "kostya")
    password = os.getenv("DESC_TOM_PASSWORD")
    client = Client.from_credentials(username, password)

    query = f'''
        SELECT * FROM elasticc_brokerclassifier ORDER BY "brokerName", "brokerVersion", "classifierName", "classifierId"
    '''
    data = client(query)
    logging.info(pformat(data))
    classifiers = {row['classifierId']: f'{row["brokerName"]} {row["brokerVersion"]} {row["classifierName"]}'
                   for row in data}

    if include_missed:
        best_last_join_type = 'LEFT'
        join_object_sent = '''
            INNER JOIN (
                SELECT
                "diaObjectId", bool_or("alertSentTimestamp" IS NOT NULL) AS "is_sent"
                    FROM elasticc_diaalert
                    GROUP BY "diaObjectId" 
                ) object_sent_record
                    ON (elasticc_diaobjecttruth."diaObjectId" = object_sent_record."diaObjectId")
                '''
        where = 'WHERE object_sent_record."is_sent"'
    else:
        best_last_join_type = 'INNER'
        join_object_sent = ''
        where = ''

    if definition == 'last_best':
        distinct_order = 'elasticc_diaalert."alertSentTimestamp", elasticc_brokerclassification."probability"'
    elif definition == 'best':
        # I think we need additional sorting over alertSentTimestamp to get the deterministic result for the case of
        # equal probabilities (I've seen prob of 1.0)
        distinct_order = 'elasticc_brokerclassification."probability", elasticc_diaalert."alertSentTimestamp"'
    else:
        raise ValueError(f'Unknown classification definition: {definition}')

    dfs = {}
    for classifier_id_, classifier_name in classifiers.items():
        if classifier_id is not None and classifier_id != classifier_id_:
            continue

        logging.info(f'Getting classifications for {classifier_name}')
        query = f'''
            SELECT best_last."classId" AS pred_class,
                   elasticc_gentypeofclassid."classId" AS true_class,
                   COUNT(*) AS n
            FROM elasticc_diaobjecttruth
            INNER JOIN elasticc_gentypeofclassid
                ON (elasticc_diaobjecttruth.gentype = elasticc_gentypeofclassid.gentype)
            {join_object_sent}
            {best_last_join_type} JOIN
            (
               SELECT DISTINCT ON (elasticc_diaalert."diaObjectId")
                  elasticc_brokerclassification."classId", elasticc_brokerclassification."probability", elasticc_diaalert."diaObjectId"
               FROM elasticc_brokerclassification
               INNER JOIN elasticc_brokermessage
                  ON elasticc_brokerclassification."brokerMessageId"=elasticc_brokermessage."brokerMessageId"
               INNER JOIN elasticc_diaalert
                  ON elasticc_brokermessage."alertId"=elasticc_diaalert."alertId"
               WHERE elasticc_brokerclassification."classifierId"={classifier_id_}
               ORDER BY elasticc_diaalert."diaObjectId", {distinct_order} DESC
            ) best_last
            ON (best_last."diaObjectId" = elasticc_diaobjecttruth."diaObjectId")
            {where}
            GROUP BY pred_class, true_class
            ORDER BY pred_class, true_class
        '''
        data = client(query)
        logging.info(pformat(data))
        if len(data) == 0:
            logging.warning(f'No data for {classifier_name}')
            continue
        df = pd.DataFrame.from_records(data)
        df['classifier_id'] = classifier_id_
        df['classifier_name'] = classifier_name
        df['pred_class'] = df['pred_class'].fillna(-1).astype(int)
        dfs[classifier_id_] = df

    return dfs


def _conf_annotation(count: int, fraction: float) -> str:
    percent = np.round(fraction * 100)
    if count < 1_000_000:
        count_str = str(count)
    else:
        count_str = f'{count:.3g}'
    return f'{percent}%\n{count_str}'


conf_annotation = np.vectorize(_conf_annotation)


def plot_matrix(matrix: pd.DataFrame, *, norm: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    from sklearn.metrics import confusion_matrix

    plt.figure(figsize=(20, 20))
    plt.gca().set_aspect(
        # aspect=len(np.unique(matrix['true_class'])) / len(np.unique(matrix['pred_class'])),
        aspect='equal',
        adjustable='box',
    )
    name = matrix.iloc[0]['classifier_name']
    counts = confusion_matrix(
        y_true=matrix['true_class'],
        y_pred=matrix['pred_class'],
        sample_weight=matrix['n'],
        normalize=None,
    )
    # Remove empty lines corresponding to missed and Other classes
    idx = np.where(np.sum(counts, axis=1) > 0)[0], np.where(np.sum(counts, axis=0) > 0)[0]
    counts = counts[idx[0], :][:, idx[1]]
    fractions = confusion_matrix(
        y_true=matrix['true_class'],
        y_pred=matrix['pred_class'],
        sample_weight=matrix['n'],
        normalize=norm,
    )[idx[0], :][:, idx[1]]
    annotations = conf_annotation(counts, fractions)
    true_labels = np.vectorize(TAXONOMY.get)(np.unique(matrix['true_class']))
    pred_labels = np.vectorize(TAXONOMY.get)(np.unique(matrix['pred_class']))
    sns.heatmap(fractions,
                cmap='Blues', vmin=0, vmax=1,
                annot=annotations, fmt='s', annot_kws={"fontsize": 10},
                xticklabels=pred_labels, yticklabels=true_labels)
    for j, label in enumerate(true_labels):
        try:
            i = np.where(pred_labels == label)[0].item()
        except ValueError:
            logging.warning(f'{label} not found in predictions for {name}')
            continue
        plt.gca().add_patch(Rectangle((i, j), 1, 1, ec='black', fc='none', lw=2))
    plt.title(name)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.savefig(f'{name}.pdf')
    plt.close()


def main(cli_args=None):
    args = parse_args(cli_args)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s' )
    dfs = get_classifications(definition=args.definition, classifier_id=args.classifier_id,
                              include_missed=args.include_missed)
    if args.save:
        df = pd.concat(list(dfs.values()))
        df.to_csv('conf_matrices.csv', index=False)
    if args.plot:
        for classifier_id, matrix in dfs.items():
            plot_matrix(matrix, norm=args.norm)



# ======================================================================
if __name__ == "__main__":
    main()


# ======================================================================
# elasticc tables as of 2022-08-05
#
# The order of the columns is what happens to be in the database, as a
# result of the specific history of django databse migrations.  It's not
# a sane order, alas.  You can find the same schema in the django source
# code, where the columns are in a more sane order; look at
# https://github.com/LSSTDESC/tom_desc/blob/main/elasticc/models.py

#             Table "public.elasticc_diaalert"
#     Column    |  Type  | Collation | Nullable | Default 
# --------------+--------+-----------+----------+---------
#  alertId      | bigint |           | not null | 
#  diaObjectId | bigint |           |          | 
#  diaSourceId | bigint |           |          | 
# Indexes:
#     "elasticc_diaalert_pkey" PRIMARY KEY, btree ("alertId")
#     "elasticc_diaalert_diaObjectId_809a8089" btree ("diaObjectId")
#     "elasticc_diaalert_diaSourceId_1f178060" btree ("diaSourceId")
# Foreign-key constraints:
#     "elasticc_diaalert_diaObjectId_809a8089_fk_elasticc_" FOREIGN KEY ("diaObjectId") REFERENCES elasticc_diaobject("diaObjectId") DEFERRABLE INITIALLY DEFERRED
#     "elasticc_diaalert_diaSourceId_1f178060_fk_elasticc_" FOREIGN KEY ("diaSourceId") REFERENCES elasticc_diasource("diaSourceId") DEFERRABLE INITIALLY DEFERRED
# Referenced by:
#     TABLE "elasticc_diaalertprvsource" CONSTRAINT "elasticc_diaalertprv_diaAlert_id_68c18d04_fk_elasticc_" FOREIGN KEY ("diaAlert_id") REFERENCES elasticc_diaalert("alertId") DEFERRABLE INITIALLY DEFERRED
#     TABLE "elasticc_diaalertprvforcedsource" CONSTRAINT "elasticc_diaalertprv_diaAlert_id_ddb308dc_fk_elasticc_" FOREIGN KEY ("diaAlert_id") REFERENCES elasticc_diaalert("alertId") DEFERRABLE INITIALLY DEFERRED
# 
#                     Table "public.elasticc_diaobject"
#         Column        |       Type       | Collation | Nullable | Default 
# ----------------------+------------------+-----------+----------+---------
#  diaObjectId          | bigint           |           | not null | 
#  ra                   | double precision |           | not null | 
#  decl                 | double precision |           | not null | 
#  mwebv                | double precision |           |          | 
#  mwebv_err            | double precision |           |          | 
#  z_final              | double precision |           |          | 
#  z_final_err          | double precision |           |          | 
#  hostgal_ellipticity  | double precision |           |          | 
#  hostgal_sqradius     | double precision |           |          | 
#  hostgal_zspec        | double precision |           |          | 
#  hostgal_zspec_err    | double precision |           |          | 
#  hostgal_zphot_q010   | double precision |           |          | 
#  hostgal_zphot_q020   | double precision |           |          | 
#  hostgal_zphot_q030   | double precision |           |          | 
#  hostgal_zphot_q040   | double precision |           |          | 
#  hostgal_zphot_q050   | double precision |           |          | 
#  hostgal_zphot_q060   | double precision |           |          | 
#  hostgal_zphot_q070   | double precision |           |          | 
#  hostgal_zphot_q080   | double precision |           |          | 
#  hostgal_zphot_q090   | double precision |           |          | 
#  hostgal_zphot_q100   | double precision |           |          | 
#  hostgal_mag_u        | double precision |           |          | 
#  hostgal_mag_g        | double precision |           |          | 
#  hostgal_mag_r        | double precision |           |          | 
#  hostgal_mag_i        | double precision |           |          | 
#  hostgal_mag_z        | double precision |           |          | 
#  hostgal_mag_Y        | double precision |           |          | 
#  hostgal_ra           | double precision |           |          | 
#  hostgal_dec          | double precision |           |          | 
#  hostgal_snsep        | double precision |           |          | 
#  hostgal_magerr_u     | double precision |           |          | 
#  hostgal_magerr_g     | double precision |           |          | 
#  hostgal_magerr_r     | double precision |           |          | 
#  hostgal_magerr_i     | double precision |           |          | 
#  hostgal_magerr_z     | double precision |           |          | 
#  hostgal_magerr_Y     | double precision |           |          | 
#  hostgal2_ellipticity | double precision |           |          | 
#  hostgal2_sqradius    | double precision |           |          | 
#  hostgal2_zphot       | double precision |           |          | 
#  hostgal2_zphot_err   | double precision |           |          | 
#  hostgal2_zphot_q010  | double precision |           |          | 
#  hostgal2_zphot_q020  | double precision |           |          | 
#  hostgal2_zphot_q030  | double precision |           |          | 
#  hostgal2_zphot_q040  | double precision |           |          | 
#  hostgal2_zphot_q050  | double precision |           |          | 
#  hostgal2_zphot_q060  | double precision |           |          | 
#  hostgal2_zphot_q070  | double precision |           |          | 
#  hostgal2_zphot_q080  | double precision |           |          | 
#  hostgal2_ellipticity | double precision |           |          | 
#  hostgal2_sqradius    | double precision |           |          | 
#  hostgal2_zphot       | double precision |           |          | 
#  hostgal2_zphot_err   | double precision |           |          | 
#  hostgal2_zphot_q010  | double precision |           |          | 
#  hostgal2_zphot_q020  | double precision |           |          | 
#  hostgal2_zphot_q030  | double precision |           |          | 
#  hostgal2_zphot_q040  | double precision |           |          | 
#  hostgal2_zphot_q050  | double precision |           |          | 
#  hostgal2_zphot_q060  | double precision |           |          | 
#  hostgal2_zphot_q070  | double precision |           |          | 
#  hostgal2_zphot_q080  | double precision |           |          | 
#  hostgal2_zphot_q090  | double precision |           |          | 
#  hostgal2_zphot_q100  | double precision |           |          | 
#  hostgal2_mag_u       | double precision |           |          | 
#  hostgal2_mag_g       | double precision |           |          | 
#  hostgal2_mag_r       | double precision |           |          | 
#  hostgal2_mag_i       | double precision |           |          | 
#  hostgal2_mag_z       | double precision |           |          | 
#  hostgal2_mag_Y       | double precision |           |          | 
#  hostgal2_ra          | double precision |           |          | 
#  hostgal2_dec         | double precision |           |          | 
#  hostgal2_snsep       | double precision |           |          | 
#  hostgal2_magerr_u    | double precision |           |          | 
#  hostgal2_magerr_g    | double precision |           |          | 
#  hostgal2_magerr_r    | double precision |           |          | 
#  hostgal2_magerr_i    | double precision |           |          | 
#  hostgal2_magerr_z    | double precision |           |          | 
#  hostgal2_magerr_Y    | double precision |           |          | 
#  simVersion           | text             |           |          | 
#  hostgal2_zphot_q000  | double precision |           |          | 
#  hostgal2_zspec       | double precision |           |          | 
#  hostgal2_zspec_err   | double precision |           |          | 
#  hostgal_zphot        | double precision |           |          | 
#  hostgal_zphot_err    | double precision |           |          | 
#  hostgal_zphot_p50    | double precision |           |          | 
#  hostgal_zphot_q000   | double precision |           |          | 
#  hostgal2_zphot_p50   | double precision |           |          | 
# Indexes:
#     "elasticc_diaobject_pkey" PRIMARY KEY, btree ("diaObjectId")
#     "idx_elasticc_diaobject_q3c" btree (q3c_ang2ipix(ra, decl))
# Referenced by:
#     TABLE "elasticc_diaalert" CONSTRAINT "elasticc_diaalert_diaObjectId_809a8089_fk_elasticc_" FOREIGN KEY ("diaObjectId") REFERENCES elasticc_diaobject("diaObjectId") DEFERRABLE INITIALLY DEFERRED
#     TABLE "elasticc_diaforcedsource" CONSTRAINT "elasticc_diaforcedso_diaObjectId_8b1bc498_fk_elasticc_" FOREIGN KEY ("diaObjectId") REFERENCES elasticc_diaobject("diaObjectId") DEFERRABLE INITIALLY DEFERRED
#     TABLE "elasticc_diaobjecttruth" CONSTRAINT "elasticc_diaobjecttr_diaObjectId_b5103ef2_fk_elasticc_" FOREIGN KEY ("diaObjectId") REFERENCES elasticc_diaobject("diaObjectId") DEFERRABLE INITIALLY DEFERRED
#     TABLE "elasticc_diasource" CONSTRAINT "elasticc_diasource_diaObjectId_3b88bc59_fk_elasticc_" FOREIGN KEY ("diaObjectId") REFERENCES elasticc_diaobject("diaObjectId") DEFERRABLE INITIALLY DEFERRED
# 
#                    Table "public.elasticc_diasource"
#       Column       |       Type       | Collation | Nullable | Default 
# -------------------+------------------+-----------+----------+---------
#  diaSourceId       | bigint           |           | not null | 
#  ccdVisitId        | bigint           |           | not null | 
#  parentDiaSourceId | bigint           |           |          | 
#  midPointTai       | double precision |           | not null | 
#  filterName        | text             |           | not null | 
#  ra                | double precision |           | not null | 
#  decl              | double precision |           | not null | 
#  psFlux            | double precision |           | not null | 
#  psFluxErr         | double precision |           | not null | 
#  snr               | double precision |           | not null | 
#  nobs              | double precision |           |          | 
#  diaObjectId      | bigint           |           |          | 
# Indexes:
#     "elasticc_diasource_pkey" PRIMARY KEY, btree ("diaSourceId")
#     "elasticc_diasource_diaObjectId_3b88bc59" btree ("diaObjectId")
#     "elasticc_diasource_midPointTai_5766b47f" btree ("midPointTai")
#     "idx_elasticc_diasource_q3c" btree (q3c_ang2ipix(ra, decl))
# Foreign-key constraints:
#     "elasticc_diasource_diaObjectId_3b88bc59_fk_elasticc_" FOREIGN KEY ("diaObjectId") REFERENCES elasticc_diaobject("diaObjectId") DEFERRABLE INITIALLY DEFERRED
# Referenced by:
#     TABLE "elasticc_diaalert" CONSTRAINT "elasticc_diaalert_diaSourceId_1f178060_fk_elasticc_" FOREIGN KEY ("diaSourceId") REFERENCES elasticc_diasource("diaSourceId") DEFERRABLE INITIALLY DEFERRED
#     TABLE "elasticc_diaalertprvsource" CONSTRAINT "elasticc_diaalertprv_diaSourceId_91fa84a3_fk_elasticc_" FOREIGN KEY ("diaSourceId") REFERENCES elasticc_diasource("diaSourceId") DEFERRABLE INITIALLY DEFERRED
# 
#                 Table "public.elasticc_diaforcedsource"
#       Column       |       Type       | Collation | Nullable | Default 
# -------------------+------------------+-----------+----------+---------
#  diaForcedSourceId | bigint           |           | not null | 
#  ccdVisitId        | bigint           |           | not null | 
#  midPointTai       | double precision |           | not null | 
#  filterName        | text             |           | not null | 
#  psFlux            | double precision |           | not null | 
#  psFluxErr         | double precision |           | not null | 
#  totFlux           | double precision |           | not null | 
#  totFluxErr        | double precision |           | not null | 
#  diaObjectId      | bigint           |           | not null | 
# Indexes:
#     "elasticc_diaforcedsource_pkey" PRIMARY KEY, btree ("diaForcedSourceId")
#     "elasticc_diaforcedsource_diaObjectId_8b1bc498" btree ("diaObjectId")
#     "elasticc_diaforcedsource_midPointTai_a80b03af" btree ("midPointTai")
# Foreign-key constraints:
#     "elasticc_diaforcedso_diaObjectId_8b1bc498_fk_elasticc_" FOREIGN KEY ("diaObjectId") REFERENCES elasticc_diaobject("diaObjectId") DEFERRABLE INITIALLY DEFERRED
# Referenced by:
#     TABLE "elasticc_diaalertprvforcedsource" CONSTRAINT "elasticc_diaalertprv_diaForcedSource_id_783ade34_fk_elasticc_" FOREIGN KEY ("diaForcedSource_id") REFERENCES elasticc_diaforcedsource("diaForcedSourceId") DEFERRABLE INITIALLY DEFERRED
# 
#                  Table "public.elasticc_diatruth"
#     Column    |       Type       | Collation | Nullable | Default 
# --------------+------------------+-----------+----------+---------
#  diaSourceId  | bigint           |           | not null | 
#  diaObjectId  | bigint           |           |          | 
#  detect       | boolean          |           |          | 
#  gentype | integer          |           |          | 
#  true_genmag  | double precision |           |          | 
#  mjd          | double precision |           |          | 
# Indexes:
#     "elasticc_diatruth_diaSourceId_648273bb_pk" PRIMARY KEY, btree ("diaSourceId")
#     "elasticc_diatruth_diaObjectId_7dd96889" btree ("diaObjectId")
#     "elasticc_diatruth_diaSourceId_648273bb_uniq" UNIQUE CONSTRAINT, btree ("diaSourceId")
# 
#                  Table "public.elasticc_diaobjecttruth"
#        Column       |       Type       | Collation | Nullable | Default 
# --------------------+------------------+-----------+----------+---------
#  libid              | integer          |           | not null | 
#  sim_searcheff_mask | integer          |           | not null | 
#  gentype            | integer          |           | not null | 
#  sim_template_index | integer          |           | not null | 
#  zcmb               | double precision |           | not null | 
#  zhelio             | double precision |           | not null | 
#  zcmb_smear         | double precision |           | not null | 
#  ra                 | double precision |           | not null | 
#  dec                | double precision |           | not null | 
#  mwebv              | double precision |           | not null | 
#  galid              | bigint           |           | not null | 
#  galzphot           | double precision |           | not null | 
#  galzphoterr        | double precision |           | not null | 
#  galsnsep           | double precision |           | not null | 
#  galsnddlr          | double precision |           | not null | 
#  rv                 | double precision |           | not null | 
#  av                 | double precision |           | not null | 
#  mu                 | double precision |           | not null | 
#  lensdmu            | double precision |           | not null | 
#  peakmjd            | double precision |           | not null | 
#  mjd_detect_first   | double precision |           | not null | 
#  mjd_detect_last    | double precision |           | not null | 
#  dtseason_peak      | double precision |           | not null | 
#  peakmag_u          | double precision |           | not null | 
#  peakmag_g          | double precision |           | not null | 
#  peakmag_r          | double precision |           | not null | 
#  peakmag_i          | double precision |           | not null | 
#  peakmag_z          | double precision |           | not null | 
#  peakmag_Y          | double precision |           | not null | 
#  snrmax             | double precision |           | not null | 
#  snrmax2            | double precision |           | not null | 
#  snrmax3            | double precision |           | not null | 
#  nobs               | integer          |           | not null | 
#  nobs_saturate      | integer          |           | not null | 
#  diaObjectId       | bigint           |           | not null | 
# Indexes:
#     "elasticc_diaobjecttruth_pkey" PRIMARY KEY, btree ("diaObjectId")
#     "elasticc_diaobjecttruth_gentype_480cd308" btree (gentype)
#     "elasticc_diaobjecttruth_sim_template_index_b33f9ab4" btree (sim_template_index)
# Foreign-key constraints:
#     "elasticc_diaobjecttr_diaObjectId_b5103ef2_fk_elasticc_" FOREIGN KEY ("diaObjectId") REFERENCES elasticc_diaobject("diaObjectId") DEFERRABLE INITIALLY DEFERRED
# 
#                                 Table "public.elasticc_gentypeofclassid"
#     Column     |  Type   | Collation | Nullable |                        Default                         
# ---------------+---------+-----------+----------+--------------------------------------------------------
#  id            | integer |           | not null | nextval('elasticc_gentypeofclassid_id_seq'::regclass)
#  classId       | integer |           | not null | 
#  gentype | integer |           |          | 
#  description   | text    |           | not null | 
# Indexes:
#     "elasticc_gentypeofclassid_pkey" PRIMARY KEY, btree (id)
#     "elasticc_gentypeofclassid_classId_27acf0d1" btree ("classId")
#     "elasticc_gentypeofclassid_gentype_63c19152" btree (gentype)
# 
# tom_desc=# \d elasticc_brokermessage
#                                                      Table "public.elasticc_brokermessage"
#           Column          |           Type           | Collation | Nullable |                             Default                              
# --------------------------+--------------------------+-----------+----------+------------------------------------------------------------------
#  brokerMessageId           | bigint                   |           | not null | nextval('"elasticc_brokermessage_brokerMessageId_seq"'::regclass)
#  streamMessageId          | bigint                   |           |          | 
#  topicName                | character varying(200)   |           |          | 
#  alertId                  | bigint                   |           | not null | 
#  diaSourceId              | bigint                   |           | not null | 
#  descIngestTimestamp      | timestamp with time zone |           | not null | 
#  elasticcPublishTimestamp | timestamp with time zone |           |          | 
#  brokerIngestTimestamp    | timestamp with time zone |           |          | 
#  modified                 | timestamp with time zone |           | not null | 
#  msgHdrTimestamp          | timestamp with time zone |           |          | 
# Indexes:
#     "elasticc_brokermessage_pkey" PRIMARY KEY, btree ("brokerMessageId")
#     "elasticc_br_alertId_b419c9_idx" btree ("alertId")
#     "elasticc_br_dbMessa_59550d_idx" btree ("brokerMessageId")
#     "elasticc_br_diaSour_ca3044_idx" btree ("diaSourceId")
#     "elasticc_br_topicNa_73f5a4_idx" btree ("topicName", "streamMessageId")
# Referenced by:
#     TABLE "elasticc_brokerclassification" CONSTRAINT "elasticc_brokerclass_brokerMessageId_b8bd04da_fk_elasticc_" FOREIGN KEY ("brokerMessageId") REFERENCES elasticc_brokermessage("brokerMessageId") DEFERRABLE INITIALLY DEFERRED
# 
#                                                    Table "public.elasticc_brokerclassifier"
#       Column       |           Type           | Collation | Nullable |                                Default                                 
# -------------------+--------------------------+-----------+----------+------------------------------------------------------------------------
#  classifierId | bigint                   |           | not null | nextval('"elasticc_brokerclassifier_classifierId_seq"'::regclass)
#  brokerName        | character varying(100)   |           | not null | 
#  brokerVersion     | text                     |           |          | 
#  classifierName    | character varying(200)   |           | not null | 
#  classifierParams  | text                     |           |          | 
#  modified          | timestamp with time zone |           | not null | 
# Indexes:
#     "elasticc_brokerclassifier_pkey" PRIMARY KEY, btree ("classifierId")
#     "elasticc_br_brokerN_38d99f_idx" btree ("brokerName", "classifierName")
#     "elasticc_br_brokerN_86cc1a_idx" btree ("brokerName")
#     "elasticc_br_brokerN_eb7553_idx" btree ("brokerName", "brokerVersion")
# Referenced by:
#     TABLE "elasticc_brokerclassification" CONSTRAINT "elasticc_brokerclass_classifierId_91d33318_fk_elasticc_" FOREIGN KEY ("classifierId") REFERENCES elasticc_brokerclassifier("classifierId") DEFERRABLE INITIALLY DEFERRED
# 
# 
#                                                        Table "public.elasticc_brokerclassification"
#         Column         |           Type           | Collation | Nullable |                                    Default                                     
# -----------------------+--------------------------+-----------+----------+--------------------------------------------------------------------------------
#  classificationId | bigint                   |           | not null | nextval('"elasticc_brokerclassification_classificationId_seq"'::regclass)
#  classId               | integer                  |           | not null | 
#  probability           | double precision         |           | not null | 
#  modified              | timestamp with time zone |           | not null | 
#  classifierId       | bigint                   |           |          | 
#  brokerMessageId          | bigint                   |           |          | 
# Indexes:
#     "elasticc_brokerclassification_pkey" PRIMARY KEY, btree ("classificationId")
#     "elasticc_brokerclassification_classifierId_91d33318" btree ("classifierId")
#     "elasticc_brokerclassification_brokerMessageId_b8bd04da" btree ("brokerMessageId")
# Foreign-key constraints:
#     "elasticc_brokerclass_classifierId_91d33318_fk_elasticc_" FOREIGN KEY ("classifierId") REFERENCES elasticc_brokerclassifier("classifierId") DEFERRABLE INITIALLY DEFERRED
#     "elasticc_brokerclass_brokerMessageId_b8bd04da_fk_elasticc_" FOREIGN KEY ("brokerMessageId") REFERENCES elasticc_brokermessage("brokerMessageId") DEFERRABLE INITIALLY DEFERRED
