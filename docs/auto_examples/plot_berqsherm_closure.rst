.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_berqsherm_closure.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_berqsherm_closure.py:


==========================================
Berquist-Sherman Disposal Rate Adjustment
==========================================

This example demonstrates the adjustment to paid amounts and closed claim
counts using the Berquist-Sherman method.  The method calculates a `disposal_rate_`
using the `report_count_estimator`.  The disposal rates of the latest diagonal
are then used to infer adjustments to the inner diagonals of both the closed
claim triangle as well as the paid amount triangle.




.. image:: /auto_examples/images/sphx_glr_plot_berqsherm_closure_001.png
    :alt: Berquist-Sherman Closure Rate Adjustments, Berquist Sherman Paid to Unadjusted Paid, Berquist Sherman Closed Count to Unadjusted Closed Count
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    Text(0.5, 0.98, 'Berquist-Sherman Closure Rate Adjustments')





|


.. code-block:: default

    import chainladder as cl
    import matplotlib.pyplot as plt


    # Load data
    triangle = cl.load_sample('berqsherm').loc['Auto']
    # Specify Berquist-Sherman model
    berq = cl.BerquistSherman(
        paid_amount='Paid', incurred_amount='Incurred',
        reported_count='Reported', closed_count='Closed',
        reported_count_estimator=cl.Chainladder())

    # Adjust our triangle data
    berq_triangle = berq.fit_transform(triangle)
    berq_cdf = cl.Development().fit(berq_triangle).cdf_
    orig_cdf = cl.Development().fit(triangle).cdf_

    # Plot data
    fig, ((ax0, ax1)) = plt.subplots(ncols=2, figsize=(15,5))
    (berq_cdf['Paid'] / orig_cdf['Paid']).T.plot(
        kind='bar', grid=True, legend=False, ax=ax0,
        title='Berquist Sherman Paid to Unadjusted Paid').set(
        xlabel='Age to Ultimate', ylabel='Paid CDF Adjustment');

    (berq_cdf['Closed'] / orig_cdf['Closed']).T.plot(
        kind='bar', grid=True, legend=False, ax=ax1,
        title='Berquist Sherman Closed Count to Unadjusted Closed Count').set(
        xlabel='Age to Ultimate', ylabel='Closed Count CDF Adjustment');
    fig.suptitle("Berquist-Sherman Closure Rate Adjustments");


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.981 seconds)


.. _sphx_glr_download_auto_examples_plot_berqsherm_closure.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_berqsherm_closure.py <plot_berqsherm_closure.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_berqsherm_closure.ipynb <plot_berqsherm_closure.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
