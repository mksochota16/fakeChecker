document.addEventListener('DOMContentLoaded', function () {
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.addEventListener('click', () => {
        sendURL()
    });
});

function asyncTabsQuery(query) {
    return new Promise(resolve => {
        chrome.tabs.query(query, resolve);
    });
}

function sleep(ms) {
    return new Promise(resolve => {
        setTimeout(resolve, ms);
    });
}

async function sendURL() {
    const tabs = await asyncTabsQuery({active: true, currentWindow: true});
    const currentURL = tabs[0].url;
    const maxScrollTime = 5; // Set the desired max_scroll_time value here

    const apiUrl = 'http://89.76.210.123:27019/check-place';
    const queryString = '?url=' + encodeURIComponent(currentURL) + '&max_scroll_time=' + maxScrollTime;
    const requestUrl = apiUrl + queryString;

    const res = await fetch(requestUrl);
    console.log(res.status);
    const data = await res.json();
    console.log(data);
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = 'Waiting for results...<br>Estimated wait time: ' + 30 + ' seconds';
    for( let i = 0; i < 30; i++) {
        resultDiv.innerHTML = 'Waiting for results...<br>Estimated wait time: ' + 30-i + ' seconds';
        await sleep(1000);
    }

    const resultsId = data.task_id;
    let isRunning = true;

    const resultsUrl = `http://89.76.210.123:27019/check-results?results_id=${resultsId}`;
    while (isRunning) {
        const resultsRes = await fetch(resultsUrl);
        const resultsData = await resultsRes.json();

        console.log(resultsData);

        if (resultsData.type === 'running') {
          await sleep(3000); // Wait for 1 second before checking again
        } else {
          isRunning = false;
          // Do something with the final results
          console.log('Final results:', resultsData);
          resultDiv.innerHTML = 'Checked '+ resultsData.fake_checker_response.number_of_reviews_scanned + ' reviews' +
              '<br>Fake percentage: ' + Math.round(resultsData.fake_checker_response.fake_percentage) + '%';
        }
    }


}


