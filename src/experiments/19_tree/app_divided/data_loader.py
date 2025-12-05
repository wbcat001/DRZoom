"""Data loading utilities for the 19_tree experiment.
Functions:
 - load_data(base_dir=None): discover and load embedding and condensed tree; build DR list and linkage when possible
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd

def _find_file_in_base(base_dir, pattern):
    if not os.path.isdir(base_dir):
        return None
    matches = glob.glob(os.path.join(base_dir, '**', pattern), recursive=True)
    return matches[0] if matches else None


def load_data(base_dir=None):
    """Discover embedding and condensed tree artifacts and construct lightweight data structures.

    This function searches a few candidate `result` directories for `embedding.npz` and
    condensed-tree pickle files. When found it will:
    - load `EMBEDDING` (numpy array) and optional `LABELS`
    - build `DR_DUMMY_DATA`: a list of dicts suitable for Plotly scatter (`x,y,label,id`)
    - build `POINT_ID_MAP`: mapping from leaf child id -> parent cluster id
    - attempt to build `Z` and `NODE_ID_MAP` (linkage-like matrix and id map)

    Returns a dict with keys: `EMBEDDING`, `LABELS`, `DR_DUMMY_DATA`, `POINT_ID_MAP`, `Z`, `NODE_ID_MAP`, `base_dir`.
    The returned `Z` is intentionally simple (numpy array) and shape/semantics may follow the
    canonical notebook conversion or an approximate fallback. Callers should treat `Z` as an
    input to `dendro` functions rather than relying on a strict format.
    """
    # candidate base dirs (relative to this module)
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, '..', '18_rapids', 'result')),
        os.path.abspath(os.path.join(here, '..', '..', '18_rapids', 'result')),
        os.path.abspath(os.path.join(os.getcwd(), 'src', 'experiments', '18_rapids', 'result')),
        os.path.abspath(os.path.join(os.getcwd(), 'src', '18_rapids', 'result')),
    ]
    if base_dir:
        candidates.insert(0, base_dir)

    _base_dir = None
    for c in candidates:
        if os.path.isdir(c):
            _base_dir = c
            break
    if _base_dir is None:
        _base_dir = candidates[0]

    EMBEDDING = None
    LABELS = None
    POINT_ID_MAP = {}
    Z = None
    NODE_ID_MAP = {}
    DR_DUMMY_DATA = []

    # embedding
    emb_path = _find_file_in_base(_base_dir, 'embedding.npz')
    if emb_path:
        try:
            npz = np.load(emb_path, allow_pickle=True)
            if 'embedding' in npz:
                EMBEDDING = npz['embedding']
            else:
                EMBEDDING = npz[npz.files[0]]
            # try data.npz for labels
            dpath = os.path.join(os.path.dirname(emb_path), 'data.npz')
            if os.path.exists(dpath):
                dnp = np.load(dpath, allow_pickle=True)
                LABELS = dnp[dnp.files[0]] if dnp.files else None
            # build DR list
            if EMBEDDING is not None:
                n = EMBEDDING.shape[0]
                if EMBEDDING.shape[1] < 2:
                    EMBEDDING = np.concatenate([EMBEDDING, np.zeros((n, 2 - EMBEDDING.shape[1]))], axis=1)
                for i in range(n):
                    if LABELS is not None:
                        try:
                            lbl = LABELS[i]
                            if np.shape(lbl) == ():
                                label_str = str(lbl)
                            else:
                                label_str = str(i)
                        except Exception:
                            label_str = str(i)
                    else:
                        label_str = str(i)
                    DR_DUMMY_DATA.append({'x': float(EMBEDDING[i, 0]), 'y': float(EMBEDDING[i, 1]), 'label': label_str, 'id': int(i)})
                print(f"Loaded embedding from {emb_path}, n={n}")
        except Exception as e:
            print(f"Failed loading embedding: {e}")

    # condensed tree
    pkl_path = _find_file_in_base(_base_dir, '*condensed*tree*.pkl') or _find_file_in_base(_base_dir, 'condensed_tree_object.pkl')
    if pkl_path:
        try:
            with open(pkl_path, 'rb') as f:
                obj = pickle.load(f)
            condensed = obj.condensed_tree_ if hasattr(obj, 'condensed_tree_') else obj

            if hasattr(condensed, 'to_pandas'):
                df = condensed.to_pandas()
            elif hasattr(condensed, '_raw_tree'):
                raw = condensed._raw_tree
                try:
                    df = pd.DataFrame(raw)
                except Exception:
                    df = None
            else:
                df = None

            if df is not None and 'child' in df.columns and 'parent' in df.columns:
                if 'child_size' in df.columns:
                    leaf_rows = df[df['child_size'] == 1]
                else:
                    leaf_rows = df
                POINT_ID_MAP = {int(r['child']): int(r['parent']) for _, r in leaf_rows.iterrows()}
                print(f"Loaded condensed tree from {pkl_path}, point_id_map size={len(POINT_ID_MAP)}")

                # Try to construct linkage using upstream function if available (import lazily to avoid cycles)
                try:
                    from .dendro.linkage import get_linkage_matrix_from_hdbscan
                    Z_mat, node_id_map = get_linkage_matrix_from_hdbscan(condensed)
                    Z = Z_mat
                    NODE_ID_MAP = node_id_map
                    print(f"Built Z using get_linkage_matrix_from_hdbscan, shape={Z_mat.shape}, node_id_map size={len(node_id_map)}")
                except Exception as e:
                    print(f"get_linkage_matrix_from_hdbscan failed: {e}. Falling back to approximate Z builder.")
                    # fallback approximate
                    try:
                        if 'lambda_val' in df.columns and 'child_size' in df.columns:
                            merge_df = df[df['child_size'] > 1]
                            node_id_map = {}
                            linkage_rows = []
                            for idx, (_, row) in enumerate(merge_df.iterrows()):
                                parent_orig = int(row['parent'])
                                child = int(row['child'])
                                lam = float(row['lambda_val']) if 'lambda_val' in row else 0.0
                                csize = int(row['child_size']) if 'child_size' in row else 1
                                node_id_map[parent_orig] = idx
                                linkage_rows.append([child, parent_orig, lam, csize])
                            if linkage_rows:
                                Z = np.array(linkage_rows)
                                NODE_ID_MAP = node_id_map
                                print(f"Built approximate Z from condensed tree, shape={Z.shape}, node_id_map size={len(node_id_map)}")
                    except Exception as e2:
                        print(f"Fallback approximate Z builder failed: {e2}")
        except Exception as e:
            print(f"Failed loading condensed tree pickle: {e}")

    return {
        'EMBEDDING': EMBEDDING,
        'LABELS': LABELS,
        'DR_DUMMY_DATA': DR_DUMMY_DATA,
        'POINT_ID_MAP': POINT_ID_MAP,
        'Z': Z,
        'NODE_ID_MAP': NODE_ID_MAP,
        'base_dir': _base_dir,
    }
