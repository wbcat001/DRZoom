class MinimumSpanningTree(object):
    def __init__(self, mst, data):
        self._mst = mst
        self._data = data

    def plot(self, axis=None, node_size=40, node_color='k',
             node_alpha=0.8, edge_alpha=0.5, edge_cmap='viridis_r',
             edge_linewidth=2, vary_line_width=True, colorbar=True):
        """Plot the minimum spanning tree (as projected into 2D by t-SNE if required).

        Parameters
        ----------

        axis : matplotlib axis, optional
               The axis to render the plot to

        node_size : int, optional
                The size of nodes in the plot (default 40).

        node_color : matplotlib color spec, optional
                The color to render nodes (default black).

        node_alpha : float, optional
                The alpha value (between 0 and 1) to render nodes with
                (default 0.8).

        edge_cmap : matplotlib colormap, optional
                The colormap to color edges by (varying color by edge
                    weight/distance). Can be a cmap object or a string
                    recognised by matplotlib. (default `viridis_r`)

        edge_alpha : float, optional
                The alpha value (between 0 and 1) to render edges with
                (default 0.5).

        edge_linewidth : float, optional
                The linewidth to use for rendering edges (default 2).

        vary_line_width : bool, optional
                Edge width is proportional to (log of) the inverse of the
                mutual reachability distance. (default True)

        colorbar : bool, optional
                Whether to draw a colorbar. (default True)

        Returns
        -------

        axis : matplotlib axis
                The axis used the render the plot.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        if self._data.shape[0] > 32767:
            warn('Too many data points for safe rendering of an minimal spanning tree!')
            return None

        if axis is None:
            axis = plt.gca()

        if self._data.shape[1] > 2:
            # Get a 2D projection; if we have a lot of dimensions use PCA first
            if self._data.shape[1] > 32:
                # Use PCA to get down to 32 dimension
                data_for_projection = PCA(n_components=32).fit_transform(self._data)
            else:
                data_for_projection = self._data.copy()

            projection = TSNE().fit_transform(data_for_projection)
        else:
            projection = self._data.copy()

        if vary_line_width:
            line_width = edge_linewidth * (np.log(self._mst.T[2].max() / self._mst.T[2]) + 1.0)
        else:
            line_width = edge_linewidth

        line_coords = projection[self._mst[:, :2].astype(int)]
        line_collection = LineCollection(line_coords, linewidth=line_width,
                                         cmap=edge_cmap, alpha=edge_alpha)
        line_collection.set_array(self._mst[:, 2].T)

        axis.add_artist(line_collection)
        axis.scatter(projection.T[0], projection.T[1], c=node_color, alpha=node_alpha, s=node_size)
        axis.set_xticks([])
        axis.set_yticks([])

        if colorbar:
            cb = plt.colorbar(line_collection, ax=axis)
            cb.ax.set_ylabel('Mutual reachability distance')

        return axis

    def to_numpy(self):
        """Return a numpy array of weighted edges in the minimum spanning tree
        """
        return self._mst.copy()

    def to_pandas(self):
        """Return a Pandas dataframe of the minimum spanning tree.

        Each row is an edge in the tree; the columns are `from`,
        `to`, and `distance` giving the two vertices of the edge
        which are indices into the dataset, and the distance
        between those datapoints.
        """
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError('You must have pandas installed to export pandas DataFrames')

        result = DataFrame({'from': self._mst.T[0].astype(int),
                            'to': self._mst.T[1].astype(int),
                            'distance': self._mst.T[2]})
        return result

    def to_networkx(self):
        """Return a NetworkX Graph object representing the minimum spanning tree.

        Edge weights in the graph are the distance between the nodes they connect.

        Nodes have a `data` attribute attached giving the data vector of the
        associated point.
        """
        try:
            from networkx import Graph, set_node_attributes
        except ImportError:
            raise ImportError('You must have networkx installed to export networkx graphs')

        result = Graph()
        for row in self._mst:
            result.add_edge(row[0], row[1], weight=row[2])

        data_dict = {index: tuple(row) for index, row in enumerate(self._data)}
        set_node_attributes(result, data_dict, 'data')

        return result
