
from huggingface_utils import initialize_huggingface
from langchain_utils import initialize_langchain

def main():
    pipeline = initialize_huggingface()
    llm_chain = initialize_langchain(pipeline)

    text1 = """
    Narzo 60x.. Such a wonderful phone at this price range.. Same as of 11x by realme.. Only camera is 50mp but more excellent than 11x as its a new camera rather than the old 64 mp one in 11x,some samples adding..
    I got it for 11250 the 6/128 version and also exchange value of 2500 for my 3.5 old infinix hot 8..so was a very good deal for the shoping festival days... Now the price is 14999..
    Camera is excellent but lacking image stability durinh video recording.. Also 4k not available.g Battery backup 5000mah is deceny, havent tried any games yt, but the antutu score is nearby 4.2 lakhs.. So definitely will give a medium experience👌.. Also about the display very good with 700nits and rolls out with latest Android 13 with realme UI 4 with 2years update and 3years security updates.. So you will get android 14 nd 15..thats a good thing😍..
    If your getting the phone below 12500 thats a very good deal whereas the same one with number series thats 11x is just 1000rs extra for the same variants with same specs and features..
    Go for it.. Its good
    """

    print(llm_chain.run(text1))

    text2 = """
    After I read Healing the Angry Brain by Ron Potter-Efron within just two days because I was so fascinated by the information presented in it, I purchased Angry All the Time and found it utterly disappointing. It is all about interpersonal relationships and although I understand at least theoretically that many people are extremely social beings, the title is definitely a misnomer, which makes me feel cheated on.
    Toward the end of Healing the Angry Brain this second book was recommended for people who suffer chronic anger, so that's what I expected. However, chronic anger has not necessarily anything to do with relationships, you can live as withdrawn as a hermit and still be angry, even for no reason at all and only in your mind by using fantasies. I feel that my low-grade, non-violent everyday anger seeks out and successfully finds topics through which it can express itself, not necessarily the other way round. Like a person suffering from depression may think negative thoughts until they find the whole world depressing, an angry person has an almost insuperable tendency to think angry thoughts until they experience the whole world as one huge offense.
    Maybe after the first book my expectations were too high. I loved all the scientific explanations and instructions on neuroplasticity and didn't find any of them in Angry All the Time. So it was boring, disappointing and unpleasant to read. Not that I would agree with someone who found the style sarcastic. It's slightly ironic at most, and style is always a matter of taste.
    Here is why I'm disappointed: The whole book describes the problem -- anger -- and its possible results. There is no background information nor anything new, and there are no solutions. If we didn't already know we have an anger problem we would most certainly not buy this kind of book. Meanwhile I have a little library on the topic, and most of them are utter BS because they either contain simple compilations of websites you can read for free, or they suggest pseudo-solutions. Once you realize you're angry it's certainly too late to interrupt the process by taking deep breaths or something similar. Getting angry means that one simple and intense emotion takes over, and one could as well try getting out of a crying fit or panic attack by counting to ten -- it's a ridiculous suggestion.
    """

    print(llm_chain.run(text2))

    text3 = """
    It was a surprise birthday party for him! He was speechless with joy when he saw the mountain of birthday presents waiting for him. All his friends crowded around him, wishing him a happy birthday.
    """

    print(llm_chain.run(text3))

    text4 = """
    Any relationship is like a rollercoaster ride, as we may feel happy sometimes and sad during others, but what is important is that you communicate your state of mind to your partner. We bring you some heart-touching sad paragraphs that you can share with your partner if you are feeling dejected but unable to convey verbally. While most of us focus on happiness, which is a good thing, being vocal and expressive about your sadness, fears, insecurities with your partner is also equally important.
    """

    print(llm_chain.run(text4))

    text5 = """
    njn valare santhoshavan aanu pakshe enik chilappol valare deshyam varum.
    """

    print(llm_chain.run(text5))

    text6 = """
    njn endhinu pokanam and she was valare pissed at me innale. So Iam not planning for any yatra. Njn alone ayit day plan cheyyum.
    """

    print(llm_chain.run(text6))

if __name__ == "__main__":
    main()
